
import numpy as np
from scipy.optimize import minimize, brentq
from src.spm_model import DualElectrodeSPM

class HybridOptimizer:
    def __init__(self, model: DualElectrodeSPM):
        self.model = model
        # 参数边界 (近似值)
        # R_f 不再辨识，改用阻抗测量值 Re
        self.bounds = [
            (1e-15, 1e-12), # D_p
            (1e-15, 1e-12), # D_n
            (1e-13, 1e-9),  # k_p
            (1e-13, 1e-9),  # k_n
            (1e-6, 20e-6),  # R_p
            (1e-6, 20e-6),  # R_n
            (30000, 60000), # c_max_p
            (20000, 50000), # c_max_n
        ]
        
        # 正则化: 上一轮辨识参数 (用于平滑约束)
        self.prev_params = None
        # 正则化强度 λ
        # 通过实验调优: λ=5e-4 在 RMSE (+1mV) 和参数平滑 (撞边界率 -46%) 间取得最佳平衡
        self.reg_lambda = 5e-4
        
    def _solve_initial_soc(self, V_target):
        """
        使用二分法求解初始 SOC，使得 OCV(SOC) = V_target
        """
        def error_func(soc):
            return self.model.compute_ocv_voltage(soc) - V_target
            
        try:
            # 假设 SOC 在 0 到 1 之间
            # 考虑到 OCV 可能不单调或有噪声，限制范围
            soc_sol = brentq(error_func, 0.0, 1.0, xtol=1e-3)
            return soc_sol
        except ValueError:
            # 如果解不在 [0, 1] 范围内 (例如电压过高或过低)
            if error_func(1.0) * error_func(0.0) > 0:
                # 同号，说明在范围外
                if abs(error_func(1.0)) < abs(error_func(0.0)):
                    return 1.0
                else:
                    return 0.0
            return 0.5 # 默认

    def objective_function(self, params, t, I, V_meas):
        # 更新模型参数
        self.model.update_params(params)
        
        R_f = getattr(self.model, 'R_f', 0.05)
        V_ocv_target = V_meas[0] + I[0] * R_f
        V_ocv_target = np.clip(V_ocv_target, 3.0, 4.3)
        
        init_soc = self._solve_initial_soc(V_ocv_target)
        self.model.reset_state(init_soc)
            
        # 模拟
        try:
            V_pred, _ = self.model.simulate_cycle(t, I)
        except Exception as e:
            return 1e6
            
        # 加权 RMSE
        weights = np.ones_like(V_meas)
        mask = (V_meas >= 3.8) & (V_meas <= 4.1)
        weights[mask] = 2.0
        
        mse = np.average((V_pred - V_meas)**2, weights=weights)
        rmse = np.sqrt(mse)
        
        # 正则化惩罚: 约束参数相对于上一轮不要跳变太大
        reg_penalty = 0.0
        if self.prev_params is not None:
            # 使用对数比值惩罚 (参数跨数量级，绝对差无意义)
            # penalty = λ * Σ (ln(p_i / p_i_prev))^2
            for i in range(len(params)):
                if params[i] > 0 and self.prev_params[i] > 0:
                    log_ratio = np.log(params[i] / self.prev_params[i])
                    reg_penalty += log_ratio ** 2
            reg_penalty *= self.reg_lambda
        
        return rmse + reg_penalty

    def pso_search(self, t, I, V_meas, n_particles=30, n_iterations=20):
        # 自适应 PSO 实现
        dim = len(self.bounds)
        
        # 初始化粒子
        positions = np.zeros((n_particles, dim))
        velocities = np.zeros((n_particles, dim))
        pbest_pos = np.zeros((n_particles, dim))
        pbest_val = np.full(n_particles, np.inf)
        gbest_pos = np.zeros(dim)
        gbest_val = np.inf
        
        # 在边界内初始化位置
        # 如果有上一轮参数，将一半粒子初始化在其附近（局部搜索）
        n_local = n_particles // 2 if self.prev_params is not None else 0
        
        for i in range(dim):
            low, high = self.bounds[i]
            # 全局随机粒子
            positions[n_local:, i] = np.random.uniform(low, high, n_particles - n_local)
            velocities[:, i] = np.random.uniform(-1, 1, n_particles) * (high - low) * 0.1
        
        if self.prev_params is not None:
            # 局部粒子: 在上一轮参数附近对数扰动 (±0.5个数量级)
            for i in range(dim):
                low, high = self.bounds[i]
                center = self.prev_params[i]
                for j in range(n_local):
                    log_perturb = np.random.normal(0, 0.3)  # ~±0.3个数量级
                    val = center * (10 ** log_perturb)
                    positions[j, i] = np.clip(val, low, high)
            
        # 自适应参数
        w_max, w_min = 0.9, 0.4
        c1_max, c1_min = 2.5, 0.5
        c2_max, c2_min = 0.5, 2.5
        
        # 循环
        for it in range(n_iterations):
            # 线性递减惯性权重
            w = w_max - (w_max - w_min) * (it / n_iterations)
            # 时变加速系数
            # c1 (认知) 减少, c2 (社会) 增加
            c1 = c1_max - (c1_max - c1_min) * (it / n_iterations)
            c2 = c2_min + (c2_max - c2_min) * (it / n_iterations)
            
            for i in range(n_particles):
                # 评估
                val = self.objective_function(positions[i], t, I, V_meas)
                
                # 更新 PBest
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest_pos[i] = positions[i].copy()
                    
                # 更新 GBest
                if val < gbest_val:
                    gbest_val = val
                    gbest_pos = positions[i].copy()
            
            # 更新速度和位置
            r1 = np.random.rand(n_particles, dim)
            r2 = np.random.rand(n_particles, dim)
            
            velocities = w * velocities + c1 * r1 * (pbest_pos - positions) + c2 * r2 * (gbest_pos - positions)
            positions = positions + velocities
            
            # 限制边界
            for d in range(dim):
                positions[:, d] = np.clip(positions[:, d], self.bounds[d][0], self.bounds[d][1])
                
        return gbest_pos

    def run(self, t, I, V_meas, prev_params=None):
        """
        运行 PSO-LBFGS 混合优化
        
        参数:
            t, I, V_meas: 时间、电流、测量电压
            prev_params: 上一轮辨识参数 (8维), 用于正则化平滑约束
        """
        # 设置正则化参考
        self.prev_params = prev_params
        
        # 1. PSO 阶段
        initial_guess = self.pso_search(t, I, V_meas, n_particles=30, n_iterations=30)
        
        # 2. L-BFGS-B 阶段
        result = minimize(
            self.objective_function, 
            initial_guess, 
            args=(t, I, V_meas), 
            method='L-BFGS-B', 
            bounds=self.bounds,
            options={'ftol': 1e-8, 'maxiter': 100}
        )
        
        # 返回纯 RMSE (不含正则化) 用于报告
        self.prev_params = None  # 临时关闭正则化
        pure_rmse = self.objective_function(result.x, t, I, V_meas)
        self.prev_params = prev_params  # 恢复
        
        return result.x, pure_rmse

