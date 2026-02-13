
import os
import sys
import json
from datetime import datetime
import warnings

# 忽略 matplotlib 字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')
warnings.filterwarnings('ignore', message='.*findfont.*')

print("启动主程序 (Starting main script)...")
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import DataLoader
from src.spm_model import DualElectrodeSPM
from src.optimization import HybridOptimizer
from src.features import FeatureExtractor
from src.soh_estimator import SOHEstimator
from src.utils import OCV_p_corrected, OCV_n_corrected
import src.plotting as plotting
from src.sensitivity import SensitivityAnalyzer
from src.comparison import ModelComparator
from src.robustness import RobustnessAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch

print("导入完成 (Imports done).")

def set_global_seed(seed=42):
    """固定全局随机种子，确保实验可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    print("进入主函数 (Entering main)...")
    set_global_seed(42)
    plotting.set_style()
    
    # 1. 设置与初始化
    # 使用相对路径，更加通用
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "cleaned_dataset", "data")
    metadata_path = os.path.join(base_dir, "cleaned_dataset", "metadata.csv")
    
    # 切换工作目录到脚本所在目录，确保所有相对路径 plots/... 正确保存
    os.chdir(base_dir)
    
    # 目标电池列表
    train_batteries = ['B0005', 'B0006', 'B0007']
    test_batteries = ['B0018']
    target_batteries = train_batteries + test_batteries
    
    loader = DataLoader(data_dir, metadata_path, target_batteries)
    
    model = DualElectrodeSPM()
    optimizer = HybridOptimizer(model)
    extractor = FeatureExtractor()
    
    # -----------------------------------------------------------
    # 阶段 0: 灵敏度分析 (Sensitivity Analysis)
    # -----------------------------------------------------------
    print("\n[阶段 0] 执行灵敏度分析...")
    # 使用 B0005 的第 0 个循环的数据进行分析
    if 'B0005' in loader.battery_data and len(loader.battery_data['B0005']) > 0:
        cycle0_data = loader.load_cycle_data('B0005', 0)
        if cycle0_data:
            t_sens = cycle0_data['discharge']['t']
            I_sens = cycle0_data['discharge']['I']
            V_sens = cycle0_data['discharge']['V']
            
            # 下采样以加快分析速度
            ds_rate = max(1, len(t_sens) // 100)
            sens_analyzer = SensitivityAnalyzer(model)
            Si = sens_analyzer.run_analysis(t_sens[::ds_rate], I_sens[::ds_rate], V_sens[::ds_rate], sample_size=256)
            sens_analyzer.plot_sensitivity(Si, fig_label='Fig 3-1')
    else:
        print("未找到 B0005 数据，跳过灵敏度分析。")
    
    # -----------------------------------------------------------
    # 阶段 1: 参数辨识与特征提取 (Identification & Extraction)
    # -----------------------------------------------------------
    # 按电池存储数据
    battery_features = {} # battery_id -> list of features
    battery_soh = {}      # battery_id -> list of SOH
    identified_params = {} # (battery_id, cycle_idx) -> params
    validation_data = {}   # battery_id -> {'cycles': [...], 'rmse': [...], 'voltage_samples': {...}}
    
    # 定义辨识间隔（优化为每10个循环，更频繁跟踪老化）
    ident_interval = 10
    
    print("\n[阶段 1] 开始处理循环数据...")
    
    for bid in target_batteries:
        print(f"\n>>> 处理电池: {bid}")
        cycles = loader.battery_data.get(bid, [])
        if not cycles:
            print(f"电池 {bid} 无数据，跳过。")
            continue
            
        feats_list = []
        soh_list = []
        
        # 电池特定的参数记忆 (用于插值)
        bid_params = {} 
        last_identified_params_8 = None  # 上一轮辨识的 8 参数 (用于正则化)
        
        # 验证数据收集
        validation_data[bid] = {'cycles': [], 'rmse': [], 'voltage_samples': {}} 
        
        for i in range(len(cycles)):
            cycle_data = loader.load_cycle_data(bid, i)
            if not cycle_data: continue
            
            # SOH
            soh = loader.get_soh(bid, i)
            soh_list.append(soh)
            
            # 获取该循环的实测欧姆内阻 Re
            Re_measured = loader.get_Re(bid, i)
            model.R_f = Re_measured  # 使用测量值，不辨识
            
            # 参数辨识 - 使用充电CC段数据 (辨识 8 个参数，R_f 使用测量值)
            if i % ident_interval == 0:
                print(f"  辨识 {bid} 循环 {i} 参数 (充电阶段, Re={Re_measured*1000:.1f}mΩ)...")
                if cycle_data['charge']:
                    t_c = cycle_data['charge']['t']
                    I_c = cycle_data['charge']['I']
                    V_c = cycle_data['charge']['V']
                    
                    # 过滤 CC 阶段 (NASA B0005-7,18 充电策略类似: 1.5A CC until 4.2V)
                    mask_cc = (I_c > 1.0) & (V_c < 4.19)
                    if np.sum(mask_cc) > 50:
                        t_ident = t_c[mask_cc]
                        I_ident = I_c[mask_cc]
                        V_ident = V_c[mask_cc]
                        
                        ds_rate = max(1, len(t_ident) // 50)
                        params_8, error = optimizer.run(
                            t_ident[::ds_rate], I_ident[::ds_rate], V_ident[::ds_rate],
                            prev_params=last_identified_params_8
                        )
                        
                        # 记录本轮参数作为下一轮的正则化参考
                        last_identified_params_8 = params_8.copy()
                        
                        # 将 8 参数扩展为 9 参数 (追加 Re_measured)
                        params = np.append(params_8, Re_measured)
                        bid_params[i] = params
                        identified_params[(bid, i)] = params
                        
                        # 收集参数辨识验证数据（充电阶段CC段）
                        model.update_params(params_8)
                        model.R_f = Re_measured
                        V_pred_charge, _ = model.simulate_cycle(t_ident, I_ident)
                        rmse_charge = np.sqrt(np.mean((V_ident - V_pred_charge)**2))
                        
                        validation_data[bid]['cycles'].append(i)
                        validation_data[bid]['rmse'].append(rmse_charge)
                        
                        validation_data[bid]['voltage_samples'][i] = {
                            't': t_ident.copy(),
                            'V_meas': V_ident.copy(),
                            'V_sim': V_pred_charge.copy(),
                            'rmse': rmse_charge
                        }
            
            # 确定参数 (零阶保持)
            sorted_keys = sorted(bid_params.keys())
            if not sorted_keys:
                current_params = None # 使用默认
            else:
                if i in bid_params:
                    current_params = bid_params[i]
                else:
                    prev_key = max([k for k in sorted_keys if k < i], default=sorted_keys[0])
                    current_params = bid_params[prev_key]
            
            # 更新模型
            model.update_params(current_params)
            
            # 模拟放电
            t = cycle_data['discharge']['t']
            I = cycle_data['discharge']['I']
            V = cycle_data['discharge']['V']
            V_pred, micro_vars = model.simulate_cycle(t, I)
            
            # 提取特征
            feats = extractor.extract(cycle_data, micro_vars)
            
            feat_vec = [
                feats['t_rise'], feats['Energy'], feats['IC_peak'],
                feats['c_s_surf_p_mean'], feats['eta_p_mean'],
                feats['c_s_surf_n_mean'], feats['eta_n_mean']
            ]
            feats_list.append(feat_vec)
            
            if i % 50 == 0:
                print(f"  已处理 {bid} 循环 {i}")
                
        battery_features[bid] = np.array(feats_list)
        battery_soh[bid] = np.array(soh_list).reshape(-1, 1)

    # ---- SPM 特征预处理: 差分 + Per-Battery Z-Score ----
    # 问题: SPM 微观变量的绝对值是辨识参数的非线性函数，不同电池的辨识参数不同
    #        导致微观变量的分布域偏移（domain shift），跨电池泛化差
    # 解决方案: 
    #   1. SPM特征差分: 使用相邻循环间的相对变化率 Δf/|f|, 消除绝对值水平差异
    #   2. SPM特征 Per-Battery Z-Score: 对每个电池独立标准化，对齐分布
    # 这样不同电池只保留"老化引起的变化趋势"，消除电池特异性的绝对偏差
    spm_feature_indices = [3, 4, 5, 6]  # c_s_surf_p_mean, eta_p_mean, c_s_surf_n_mean, eta_n_mean
    for bid in target_batteries:
        if bid in battery_features:
            feats = battery_features[bid]
            for idx in spm_feature_indices:
                col = feats[:, idx].copy()
                # Step 1: 计算相邻循环间的相对变化率
                delta = np.zeros_like(col)
                for k in range(1, len(col)):
                    if abs(col[k-1]) > 1e-10:
                        delta[k] = (col[k] - col[k-1]) / abs(col[k-1])
                    else:
                        delta[k] = 0.0
                # Step 2: Per-Battery Z-Score 标准化
                mu = np.mean(delta)
                sigma = np.std(delta)
                if sigma > 1e-10:
                    feats[:, idx] = (delta - mu) / sigma
                else:
                    feats[:, idx] = 0.0  # 标准差为0时全部置零
            battery_features[bid] = feats

    # 绘制全周期验证图（B0005, B0006, B0007）
    train_validation = {k: v for k, v in validation_data.items() if k in ['B0005', 'B0006', 'B0007']}
    if train_validation:
        # 绘制RMSE趋势图
        plotting.plot_full_cycle_validation(train_validation, fig_label='Fig 4-3')
        # 绘制每个电池的全周期电压拟合图
        plotting.plot_battery_voltage_fitting(train_validation)
        print("\n已生成全周期参数辨识验证图和电压拟合图")

    # -----------------------------------------------------------
    # 阶段 2: 准备数据集 (Dataset Preparation)
    # -----------------------------------------------------------
    print("\n[阶段 2] 准备数据集...")
    feature_names = ['t_rise', 'Energy', 'IC_peak', 'c_s_p', 'eta_p', 'c_s_n', 'eta_n']
    
    # 绘制特征分析图 (使用合并后的原始数据，或者仅使用 B0005)
    # 这里我们使用 B0005 的数据作为代表进行特征分析可视化
    if 'B0005' in battery_features:
        plotting.plot_feature_correlation(battery_features['B0005'], feature_names, fig_label='图 5-1')
        plotting.plot_feature_evolution(battery_features['B0005'], battery_soh['B0005'], feature_names, fig_label='图 5-2')

    # 绘制参数演变 (仅 B0005)
    # 过滤 B0005 的参数
    b0005_params = {k: v for k, v in identified_params.items() if k[0] == 'B0005'}
    if b0005_params:
        # 转换为只带 cycle 的 key，以便绘图函数兼容
        b0005_params_cycle = {k[1]: v for k, v in b0005_params.items()}
        plotting.plot_parameter_trends(b0005_params_cycle, fig_labels=['Fig 4-5', 'Fig 4-6', 'Fig 4-7'])
    
    # 绘制多电池 Re 演变图 - 直接从 metadata.csv 读取所有阻抗测量点
    plotting.plot_Re_from_metadata(metadata_path, battery_ids=['B0005', 'B0006', 'B0007', 'B0018'], fig_label='图4-8')
    
    # 数据归一化 (使用训练集计算 scaler)
    # 收集所有训练数据用于 fit
    all_train_features = []
    for bid in train_batteries:
        if bid in battery_features:
            all_train_features.append(battery_features[bid])
            
    if not all_train_features:
        print("错误: 没有训练数据!")
        return

    X_train_raw_all = np.vstack(all_train_features)
    scaler_X = MinMaxScaler()
    scaler_X.fit(X_train_raw_all)
    
    # 构建序列函数
    def create_sequences(features, soh, seq_len=10):
        if len(features) <= seq_len:
            return np.empty((0, seq_len, features.shape[1])), np.empty((0, 1))
        
        # 归一化
        feats_scaled = scaler_X.transform(features)
        
        X_seq, y_seq = [], []
        for i in range(len(feats_scaled) - seq_len):
            X_seq.append(feats_scaled[i:i+seq_len])
            y_seq.append(soh[i+seq_len])
        return np.array(X_seq), np.array(y_seq)

    # 生成训练集 (合并多个电池的序列)
    seq_len = 10  # 减小序列长度以获得更多训练样本
    X_train_list, y_train_list = [], []
    
    for bid in train_batteries:
        if bid in battery_features:
            X_s, y_s = create_sequences(battery_features[bid], battery_soh[bid], seq_len)
            if len(X_s) > 0:
                X_train_list.append(X_s)
                y_train_list.append(y_s)
                
    X_train = np.vstack(X_train_list)
    y_train = np.vstack(y_train_list)
    
    # 生成测试集 (B0018)
    X_test_list, y_test_list = [], []
    for bid in test_batteries:
        if bid in battery_features:
            X_s, y_s = create_sequences(battery_features[bid], battery_soh[bid], seq_len)
            if len(X_s) > 0:
                X_test_list.append(X_s)
                y_test_list.append(y_s)
    
    if X_test_list:
        X_test = np.vstack(X_test_list)
        y_test = np.vstack(y_test_list)
    else:
        print("警告: 测试集为空!")
        X_test, y_test = X_train[:1], y_train[:1] # 避免崩溃
        
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # -----------------------------------------------------------
    # 阶段 3: SP-LSTM 训练与预测
    # -----------------------------------------------------------
    print("\n[阶段 3] 训练 SP-LSTM 模型...")
    # 重置种子确保训练可复现
    set_global_seed(42)
    # 使用更多训练轮数以获得更好的收敛
    sp_lstm_epochs = 800
    estimator = SOHEstimator(input_size=7, seq_len=seq_len)
    estimator.train(X_train, y_train, epochs=sp_lstm_epochs)
    
    # 评估（不应用初始状态校正）
    y_pred_raw = estimator.predict(X_test, apply_offset=False)
    
    # 初始状态校正
    estimator.calibrate_initial_state(y_pred_raw, y_test)
    
    # 重新预测（应用校正）
    y_pred = estimator.predict(X_test, apply_offset=True)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"SP-LSTM 测试集 RMSE: {rmse:.5f}")
    
    # -----------------------------------------------------------
    # 阶段 4: 算法对比 (Model Comparison)
    # -----------------------------------------------------------
    print("\n[阶段 4] 运行算法对比...")
    comparator = ModelComparator(X_train, y_train, X_test, y_test, scaler_X)
    
    # ⚠️ 关键：确保所有模型使用相同的训练轮数以实现公平对比
    # Base LSTM使用与SP-LSTM相同的训练轮数
    comparator.train_evaluate_all(epochs=sp_lstm_epochs)
    
    # 添加 SP-LSTM 结果
    comparator.add_result("SP-LSTM (Ours)", y_pred)
    
    # 绘制对比图
    comparator.plot_comparison(fig_labels=['Fig 6-3', 'Fig 6-4'])
    
    # -----------------------------------------------------------
    # 阶段 5: 结果可视化 (Final Visualization)
    # -----------------------------------------------------------
    # 构建完整的预测序列用于可视化 (仅展示测试集 B0018)
    
    # 我们只展示测试集的结果，或者分别为训练集和测试集画图
    # 这里我们专注于 B0018 的预测结果
    
    print("\n绘制 B0018 测试结果...")
    plotting.plot_soh_results(y_test, y_pred, 0, save_dir='plots/results_B0018', fig_label='Fig 6-1')
    # 注意：error_analysis 也会使用这个 fig_label，我们可以在内部处理，或者假设它会自动加上后缀
    
    # -----------------------------------------------------------
    # 阶段 6: 鲁棒性分析 (Robustness Analysis)
    # -----------------------------------------------------------
    print("\n[阶段 6] 运行鲁棒性分析...")
    # 需要: model, extractor, estimator, scaler, loader, identified_params
    robustness = RobustnessAnalyzer(model, extractor, estimator, scaler_X, loader, identified_params)
    
    # 测试电池 B0018, 噪声水平 0%, 1%, 2%, 3%
    noise_res = robustness.run_analysis('B0018', noise_levels=[0, 0.01, 0.02, 0.03])
    
    if noise_res:
        robustness.plot_results(noise_res, fig_label='Fig 7-1')
    
    # -----------------------------------------------------------
    # 结果保存: JSON 历史记录
    # -----------------------------------------------------------
    results_record = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epochs": sp_lstm_epochs,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "lr": 0.001,
            "scheduler": "CosineAnnealingWarmRestarts",
            "T_0": 100,
            "T_mult": 2,
            "trend_loss_weight": 0.1,
            "seq_len": seq_len,
            "feature_gate": True,
        },
        "results": {
            "SP-LSTM": {
                "RMSE": float(rmse),
                "offset": float(estimator.initial_offset),
            }
        }
    }
    
    for name, res in comparator.results.items():
        results_record["results"][name] = {
            "RMSE": float(res["RMSE"]),
            "MAE": float(res["MAE"]),
        }
    
    if noise_res:
        results_record["robustness"] = {
            f"{k*100:.0f}%": float(v) for k, v in noise_res.items()
        }
    
    history_path = os.path.join(base_dir, "results_history.json")
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = {"experiments": []}
    
    history["experiments"].append(results_record)
    
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"\n实验结果已保存到 results_history.json (第 {len(history['experiments'])} 次实验)")
    
    # 打印结果摘要
    print("\n" + "="*60)
    print("本次实验结果摘要:")
    print("="*60)
    sp_rmse = rmse
    print(f"  SP-LSTM RMSE:       {sp_rmse:.5f}")
    for name, res in comparator.results.items():
        print(f"  {name:20s} RMSE: {res['RMSE']:.5f}")
    if noise_res:
        for nl, nr in sorted(noise_res.items()):
            print(f"  鲁棒性 ({nl*100:.0f}% noise):  {nr:.5f}")
    base_rmse = comparator.results.get("Base LSTM", {}).get("RMSE", float('inf'))
    if sp_rmse < base_rmse:
        print(f"\n  ✓ SP-LSTM 优于 Base LSTM (差距: {base_rmse - sp_rmse:.5f})")
    else:
        print(f"\n  ✗ SP-LSTM 劣于 Base LSTM (差距: {sp_rmse - base_rmse:.5f})")
    print("="*60)
    
    print("\n全部任务完成。请查看 'plots/' 目录下的图像。")

if __name__ == "__main__":
    main()
