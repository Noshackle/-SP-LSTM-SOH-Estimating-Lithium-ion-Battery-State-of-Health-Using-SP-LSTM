
import torch
import torch.nn as nn
import numpy as np

class SP_LSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        """
        优化架构以提升拟合精度：
        - hidden_size: 64 (平衡容量与正则化)
        - num_layers: 2 (保持简单)
        - dropout: 0.2 (适度正则化)
        - 添加BatchNorm提升稳定性
        - 使用双向LSTM捕获前后关系
        """
        super(SP_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 特征门控: 学习各输入特征的重要性权重
        # SP-LSTM(7维)可抑制噪声SPM特征, Base LSTM(3维)保持恒等
        self.feature_scale = nn.Parameter(torch.ones(input_size))
        
        # 双向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        
        # 输出层（双向LSTM输出维度翻倍）
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        # 特征门控: 自适应加权各维特征
        x = x * self.feature_scale
        
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # 双向LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # 取最后时间步（双向拼接后的特征）
        
        # BatchNorm + Dropout + 线性映射
        out = self.bn(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class SOHEstimator:
    def __init__(self, input_size=7, seq_len=10):
        """
        优化超参数：
        - seq_len: 15→10 (减小以获得更多训练样本)
        - hidden_size: 64 (平衡容量)
        - dropout: 0.2 (适度正则化)
        - 使用双向LSTM
        """
        self.model = SP_LSTM(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2)
        self.seq_len = seq_len
        self.criterion = nn.MSELoss()
        # Adam优化器 + 适度权重衰减
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-5)
        # CosineAnnealingWarmRestarts: 周期性LR重启，帮助跳出局部最优
        # T_0=100: 首周期100 epoch, T_mult=2: 每次重启周期翻倍
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2, eta_min=1e-6
        )
        self.scaler = None
        self.initial_offset = 0.0  # 用于初始状态校正
        
    def train(self, X, y, epochs=800, verbose=True):
        """
        改进训练策略：
        - CosineAnnealingWarmRestarts周期性LR重启
        - 趋势损失权重0.1 + 特征门控
        - Early stopping (patience=200) + 模型检查点
        """
        self.model.train()
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        best_loss = float('inf')
        patience = 200
        patience_counter = 0
        best_state = None  # 模型检查点: 保存最优参数
        
        for epoch in range(epochs):
            outputs = self.model(X_tensor)
            
            # 主损失：MSE
            mse_loss = self.criterion(outputs, y_tensor)
            
            # 趋势损失：确保预测趋势与真实趋势一致
            if len(outputs) > 1:
                pred_diff = outputs[1:] - outputs[:-1]
                true_diff = y_tensor[1:] - y_tensor[:-1]
                trend_loss = torch.mean((pred_diff - true_diff) ** 2)
                loss = mse_loss + 0.1 * trend_loss  # 降低权重释放学习自由度
            else:
                loss = mse_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # CosineAnnealingWarmRestarts按epoch步进
            self.scheduler.step()
            
            # Early stopping with model checkpoint
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}, best loss: {best_loss:.6f}')
                break
            
            if verbose and (epoch+1) % 20 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, LR: {current_lr:.6f}')
        
        # 恢复最优模型参数
        if best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f'已恢复最优模型参数 (best loss: {best_loss:.6f})')
                
    def predict(self, X, apply_offset=True):
        """
        预测SOH，支持初始状态校正
        
        Args:
            X: 输入特征序列
            apply_offset: 是否应用初始状态校正
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
        
        predictions = outputs.numpy()
        
        # 应用初始状态校正
        if apply_offset and self.initial_offset != 0.0:
            predictions = predictions + self.initial_offset
            
        return predictions
    
    def calibrate_initial_state(self, y_pred, y_true):
        """
        校正初始状态，对齐预测的起始值
        
        Args:
            y_pred: 预测的SOH值
            y_true: 真实的SOH值
        """
        if len(y_pred) > 0 and len(y_true) > 0:
            # 计算初始偏移量（使用前几个样本的平均偏差）
            n_init = min(5, len(y_pred))
            self.initial_offset = np.mean(y_true[:n_init] - y_pred[:n_init])
            print(f"初始状态校正偏移量: {self.initial_offset:.6f}")
