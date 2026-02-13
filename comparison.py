
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.soh_estimator import SOHEstimator
import torch

class ModelComparator:
    def __init__(self, X_train, y_train, X_test, y_test, scaler):
        """
        X_train, X_test: 形状为 (samples, seq_len, features) 的序列数据
        y_train, y_test: 形状为 (samples, 1) 的标签
        scaler: 用于反归一化或处理特征的缩放器
        """
        self.X_train_seq = X_train
        self.y_train = y_train
        self.X_test_seq = X_test
        self.y_test = y_test
        self.scaler = scaler
        
        # 展平时序特征用于传统机器学习模型 (SVR, RF)
        # 保留时序信息：将 (samples, seq_len, 3) 展平为 (samples, seq_len*3)
        # 这样SVR和RF也能利用时序信息，确保公平对比
        n_samples_train, seq_len, n_features = X_train.shape
        self.X_train_flat = X_train[:, :, :3].reshape(n_samples_train, -1)  # (samples, seq_len*3)
        
        n_samples_test = X_test.shape[0]
        self.X_test_flat = X_test[:, :, :3].reshape(n_samples_test, -1)     # (samples, seq_len*3)
        
        # 结果字典
        self.results = {}
        
    def train_evaluate_all(self, epochs=500):
        """
        训练所有对比模型，确保公平对比：
        - 控制变量: SP-LSTM用7维特征, 其余用3维外部特征
        - 统一条件: 所有模型使用相同的初始状态校正策略（前N点均值）
        - 无额外处理: 不对任何模型做选择性平滑
        """
        print("开始训练对比模型...")
        
        # 1. SVR (支持向量回归)
        print("训练 SVR (3个外部特征×序列长度)...")
        svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.01)
        svr.fit(self.X_train_flat, self.y_train.ravel())
        y_pred_svr = svr.predict(self.X_test_flat).reshape(-1, 1)
        # 应用与SP-LSTM相同的初始状态校正（前N点均值）
        y_pred_svr = self._apply_initial_correction(y_pred_svr, "SVR")
        self._log_result("SVR", self.y_test, y_pred_svr)
        
        # 2. Random Forest (随机森林)
        print("训练 Random Forest (3个外部特征×序列长度)...")
        rf = RandomForestRegressor(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        )
        rf.fit(self.X_train_flat, self.y_train.ravel())
        y_pred_rf = rf.predict(self.X_test_flat).reshape(-1, 1)
        # 应用与SP-LSTM相同的初始状态校正
        y_pred_rf = self._apply_initial_correction(y_pred_rf, "Random Forest")
        self._log_result("Random Forest", self.y_test, y_pred_rf)
        
        # 3. Base LSTM - 独立训练 (仅使用3个外部特征)
        # 控制变量: 使用与SP-LSTM相同的LSTM架构和训练策略，但只输入3维外部特征
        # 这直接衡量 SPM 特征的增量贡献
        print("训练 Base LSTM (3个外部特征)...")
        torch.manual_seed(42)
        np.random.seed(42)
        X_train_base = self.X_train_seq[:, :, :3]  # 只取前3个外部特征
        X_test_base = self.X_test_seq[:, :, :3]
        estimator_base = SOHEstimator(input_size=3, seq_len=self.X_train_seq.shape[1])
        estimator_base.train(X_train_base, self.y_train, epochs=epochs, verbose=False)
        y_pred_base_raw = estimator_base.predict(X_test_base, apply_offset=False)
        estimator_base.calibrate_initial_state(y_pred_base_raw, self.y_test)
        y_pred_base = estimator_base.predict(X_test_base, apply_offset=True)
        self._log_result("Base LSTM", self.y_test, y_pred_base)
    
    def _apply_initial_correction(self, y_pred, model_name=""):
        """
        应用初始状态校正机制（与SP-LSTM的calibrate_initial_state完全一致）
        使用前N个样本的平均偏差进行校正
        """
        if len(y_pred) > 0 and len(self.y_test) > 0:
            n_init = min(5, len(y_pred))
            initial_offset = np.mean(self.y_test[:n_init] - y_pred[:n_init])
            y_pred_corrected = y_pred + initial_offset
            print(f"  -> {model_name} 初始状态校正偏移量: {initial_offset:.6f}")
            return y_pred_corrected
        return y_pred
    
    def _log_result(self, name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        self.results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'y_pred': y_pred
        }
        print(f"模型: {name} | RMSE: {rmse:.5f} | MAE: {mae:.5f}")

    def add_result(self, name, y_pred):
        """手动添加外部模型结果 (如 SP-LSTM)"""
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        self.results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'y_pred': y_pred
        }
        
    def plot_comparison(self, save_dir='plots/comparison', fig_labels=None):
        """
        绘制算法对比图
        fig_labels: list of 2 strings, e.g. ['图 6-3', '图 6-4']
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if not fig_labels or len(fig_labels) < 2:
            fig_labels = ['', '']
            
        # 1. 柱状图对比 RMSE
        names = list(self.results.keys())
        rmses = [self.results[n]['RMSE'] for n in names]
        
        plt.figure(figsize=(10, 6))
        # 使用 viridis 色系，SP-LSTM 用特别颜色
        n = len(names)
        colors_list = plt.cm.viridis(np.linspace(0.2, 0.7, n-1))
        colors = list(colors_list) + [plt.cm.viridis(0.95)]  # 最后一个 (SP-LSTM) 用亮色
        bars = plt.bar(names, rmses, color=colors)
        
        plt.ylabel('RMSE [dimensionless]', fontweight='bold')
        title = 'Performance Comparison of SOH Estimation Algorithms'
        if fig_labels[1]: title = f"{fig_labels[1]} {title}"
        plt.title(title, fontweight='bold')
        
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom')
                     
        plt.savefig(os.path.join(save_dir, 'model_rmse_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. SOH 预测曲线对比
        plt.figure(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(names)+1))
        plt.plot(self.y_test, 'o-', color=colors[0], linewidth=2, markersize=4, 
                 markevery=max(1, len(self.y_test)//20), label='Ground Truth')
        
        markers_list = ['s', '^', 'D', 'v']
        for i, name in enumerate(names):
            marker = markers_list[i % len(markers_list)]
            lw = 3 if 'SP-LSTM' in name else 2
            ms = 5 if 'SP-LSTM' in name else 4
            plt.plot(self.results[name]['y_pred'], marker=marker, linestyle='--', 
                     color=colors[i+1], linewidth=lw, markersize=ms, 
                     markevery=max(1, len(self.y_test)//20), label=name, alpha=0.85)
            
        plt.xlabel('Test Samples', fontweight='bold')
        plt.ylabel('State of Health (SOH)', fontweight='bold')
        
        title = 'SOH Estimation Curve Comparison'
        if fig_labels[0]: title = f"{fig_labels[0]} {title}"
        plt.title(title, fontweight='bold')
        
        plt.legend(loc='upper right', frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        fname = 'soh_curve_comparison.png'
        if fig_labels[0]:
            try:
                chap = fig_labels[0].split('-')[0].split()[-1]
                fname = f"第{chap}章_{fig_labels[0].replace(' ', '')}_SOH曲线对比.png"
            except: pass
            
        plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()
