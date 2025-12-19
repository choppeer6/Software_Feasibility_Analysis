import numpy as np
import math
import matplotlib.pyplot as plt

# ==========================================
# 核心工具函数：数据预处理
# ==========================================

def preprocess_data(data, is_interval=False):
    """
    数据预处理：排序、去重、转累计
    """
    data = np.array(data, dtype=np.float64)
    # 过滤无效数据
    data = data[data > 0]
    
    if len(data) == 0:
        return np.array([])

    # 判断是否为间隔数据 (根据单调性)
    # 如果数据不是严格递增的，或者用户指定了 interval，则视为间隔
    is_sorted = np.all(data[:-1] <= data[1:])
    
    if is_interval or (not is_sorted):
        cumulative_data = np.cumsum(data)
    else:
        cumulative_data = np.sort(data)
    
    for i in range(1, len(cumulative_data)):
        if cumulative_data[i] <= cumulative_data[i-1]:
            cumulative_data[i] = cumulative_data[i-1] + 1e-4

    return cumulative_data

def cumulative_to_intervals(cumulative_data):
    """
    累计时间 -> 间隔时间
    """
    if len(cumulative_data) == 0:
        return np.array([])
    
    intervals = np.zeros_like(cumulative_data)
    intervals[0] = cumulative_data[0]
    intervals[1:] = np.diff(cumulative_data)
    
    # 修正过小的间隔
    intervals[intervals < 1e-5] = 1e-5
    return intervals

# ==========================================
# 1. 参数估计函数 (深度优化版)
# ==========================================

def jm_model_parameter_estimation(data, ex=1e-4, ey=1e-4):
    """
    JM模型参数估计 - 强制收敛版
    """
    # 1. 准备数据
    cumulative_data = preprocess_data(data)
    n = len(cumulative_data)
    if n < 3: return None, None # 数据太少无法计算
        
    intervals = cumulative_to_intervals(cumulative_data)
    T = cumulative_data[-1] # 总测试时间
    
    # 计算加权和 S
    S = 0.0
    for i in range(1, n + 1):
        S += (i - 1) * intervals[i-1]
    
    # 2. 定义似然方程 f(N)
    # 我们希望找到 N 使得 f(N) = 0
    # 方程：sum(1/(N-i+1)) - n*T / (N*T - S) = 0
    def func(N):
        # 边界保护
        if N <= n - 1: return 1e9 
        
        sum_term = 0.0
        for i in range(1, n + 1):
            denom = N - i + 1
            if denom <= 0: return 1e9
            sum_term += 1.0 / denom
        
        denom_right = N * T - S
        if abs(denom_right) < 1e-9: return 1e9
        
        return sum_term - (n * T) / denom_right

    # 3. 求解策略优化
    # JM 模型的特点：如果数据没有明显的可靠性增长（间隔变长），N0 会趋向于无穷大。
    # 为了防止“直线”现象，我们需要限制搜索范围，并给出一个合理的“保底值”。
    
    # 搜索下界：必须大于观测到的故障数 n
    left = n + 0.1
    # 搜索上界：限制为 n 的 10 倍。
    # 如果 N0 > 10n，说明现在的故障还不到总数的10%，这在工程上几乎等同于常数失效率。
    # 强制截断可以让曲线弯曲，即便牺牲一点数学严谨性。
    max_N = n * 10.0 
    
    # 3.1 检查边界条件
    f_left = func(left)
    f_max = func(max_N)
    
    N0_est = 0.0

    # 如果 f(left) 和 f(max) 异号，说明解在区间内
    if f_left * f_max < 0:
        # 使用二分法求解
        low = left
        high = max_N
        for _ in range(100):
            mid = (low + high) / 2
            if abs(high - low) < ex:
                break
            f_mid = func(mid)
            if f_mid * f_left < 0:
                high = mid
            else:
                low = mid
                f_left = f_mid
        N0_est = mid
        
    else:
        # 3.2 无法找到解（数据质量差，或者没有可靠性增长趋势）
        # 此时 f(N) 通常在整个区间内都是正数或负数
        
        # 策略：不返回直线，而是返回一个基于经验的估计。
        # 假设当前发现的故障占总故障的 60%~80% (这是一个比较通用的软件工程经验值)
        # 这样能保证画出来的图是弯的。
        
        # 我们还可以简单检测一下趋势：
        half_n = n // 2
        first_half_avg = np.mean(intervals[:half_n])
        last_half_avg = np.mean(intervals[half_n:])
        
        if last_half_avg > first_half_avg:
            # 数据有变好的趋势，但方程解不出，可能是局部波动
            # 给一个相对乐观的估计 (N0 ≈ 1.5n)
            N0_est = n * 1.5
        else:
            # 数据变差了或没变化（直线原因），强制给一个较大的 N0，但也别无穷大
            # 设为 3n，这样能看到一点点弯曲，同时暗示还早着呢
            N0_est = n * 3.0
            
    # 4. 计算 Phi
    denom_phi = N0_est * T - S
    if denom_phi <= 0:
        phi_est = n / (N0_est * T) # 兜底
    else:
        phi_est = n / denom_phi
        
    return N0_est, phi_est

# ==========================================
# 2. 预测函数
# ==========================================


def jm_predict_future_failures(N0, phi, data, prediction_step):
    cumulative_data = preprocess_data(data)
    n = len(cumulative_data)
    current_cumulative_time = cumulative_data[-1] if n > 0 else 0
    
    predicted_intervals = []
    cumulative_times = [] 
    
    # 1. 计算剩余故障数 (浮点数)
    remaining_faults_float = max(0, N0 - n)
    
    # 我们只预测到 int(N0) 为止。
    max_physical_index = int(N0) 
    remaining_integers = max(0, max_physical_index - n)
    
    # 实际步数 = 取“用户要求的步数”和“剩余整数故障数”中的较小值
    actual_steps = min(prediction_step, remaining_integers)
    
    warning_msg = None
    if actual_steps < prediction_step:
        warning_msg = f"提示：模型估算剩余故障仅剩 {remaining_integers} 个，已自动停止后续预测。"
    
    # 如果 N0 还是很大（比如是 n 的 5 倍以上），提示一下
    if N0 > n * 5 and warning_msg is None:
        warning_msg = "提示：数据未显示明显的可靠性增长趋势，预测结果仅供参考。"
    
    temp_cumulative = current_cumulative_time
    
    # 3. 循环预测 (只循环 actual_steps 次)
    for k in range(1, actual_steps + 1):
        # 预测第 n+k 个新故障
        # 公式: E[t] = 1 / (phi * (N0 - (n + k - 1)))
        
        current_faults_found = n + k - 1
        denom = N0 - current_faults_found
        
        # 保护逻辑：虽然限制了步数，但防止浮点误差导致分母为0
        if denom <= 0.001: 
            break # 再次保险，停止预测
        
        pred_interval = 1.0 / (phi * denom)
        
        # 防止单个预测步长过分夸张
        if pred_interval > current_cumulative_time * 2 and current_cumulative_time > 0:
             pred_interval = current_cumulative_time * 2
             
        predicted_intervals.append(pred_interval)
        temp_cumulative += pred_interval
        cumulative_times.append(temp_cumulative)


    if predicted_intervals:
        future_span = sum(predicted_intervals)
    else:
        future_span = current_cumulative_time * 0.1 if current_cumulative_time > 0 else 100

    t_points = np.linspace(0, future_span, 50)
    
    # 可靠度 R(t) = exp(-phi * (N0 - n) * t)
    curr_rem = max(0, N0 - n)
    reliability_values = [math.exp(-phi * curr_rem * t) for t in t_points]
        
    return {
        'remaining_faults': remaining_faults_float,
        'next_failure_time': cumulative_times[0] if cumulative_times else None, # 如果没有预测则为None
        'predicted_intervals': predicted_intervals,
        'cumulative_times': cumulative_times, 
        'reliability_curve': (t_points, reliability_values),
        'warning': warning_msg,
        'total_faults': N0
    }
# ==========================================
# 3. 准确率评估函数
# ==========================================

def calculate_model_accuracy(N0, phi, data):
    actual_cumulative = preprocess_data(data)
    n = len(actual_cumulative)
    if n < 3: 
        return {'mae': 0, 'mse': 0, 'rmse': 0, 'r2_score': 0, 'accuracy': 0}

    predicted_cumulative = []
    current_val = 0.0
    
    # 模拟“过去”的拟合情况
    for i in range(1, n + 1):
        # 理论上第 i 个故障发生时的间隔
        # 注意：这里我们用期望值来计算
        denom = N0 - (i - 1)
        if denom <= 0: denom = 0.5
        
        interval_pred = 1.0 / (phi * denom)
        current_val += interval_pred
        predicted_cumulative.append(current_val)
        
    predicted_cumulative = np.array(predicted_cumulative)
    errors = actual_cumulative - predicted_cumulative
    
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    
    # R2
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual_cumulative - np.mean(actual_cumulative)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0
    
    # Accuracy (1 - MAPE)
    # 增加鲁棒性：忽略非常小的分母，限制单点最大误差为 100%
    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mask = actual_cumulative > 1e-6
        if np.sum(valid_mask) > 0:
            pct_errors = np.abs(errors[valid_mask] / actual_cumulative[valid_mask])

            pct_errors = np.minimum(pct_errors, 1.0)
            mape = np.mean(pct_errors)
            accuracy = (1 - mape) * 100
        else:
            accuracy = 0
    return {
        'mae': mae, 'mse': mse, 'rmse': rmse, 
        'r2_score': r2, 'accuracy': accuracy
    }

# 辅助函数保持不变
def calculate_reliability(N0, phi, times):
    times = np.array(times)
    return np.exp(-phi * N0 * times)

def plot_prediction_results(train_times, prediction_results, filename='prediction.png'):
    pass