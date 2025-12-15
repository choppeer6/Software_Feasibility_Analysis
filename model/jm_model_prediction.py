import numpy as np
import math
import matplotlib.pyplot as plt

# ==========================================
# 核心工具函数：数据预处理
# ==========================================

def preprocess_data(data, is_interval=False):
    """
    处理输入数据，确保将其转换为模型所需的【累计失效时间】
    """
    data = np.array(data, dtype=np.float64)
    if len(data) == 0:
        return np.array([])

    # 智能检测：如果数据不是单调递增的，极有可能是间隔数据，或者乱序数据
    # 如果用户没有显式指定 is_interval，我们尝试推断
    is_sorted = np.all(data[:-1] <= data[1:])
    
    # 如果明确指定是间隔数据，或者数据明显未排序（累计时间必须排序），则视为间隔数据进行累加
    if is_interval or (not is_sorted and np.all(data > 0)):
        # 将间隔转为累计: [9, 12, 11] -> [9, 21, 32]
        cumulative_data = np.cumsum(data)
        return cumulative_data
    
    # 否则假设是累计数据，进行排序确保单调性
    return np.sort(data)

def cumulative_to_intervals(cumulative_data):
    """
    将累计失效时间转换为失效间隔时间 (用于参数估计内部计算)
    输入: [9, 21, 32]
    输出: [9, 12, 11]
    """
    if len(cumulative_data) == 0:
        return np.array([])
    
    intervals = np.zeros_like(cumulative_data)
    intervals[0] = cumulative_data[0]
    intervals[1:] = np.diff(cumulative_data)
    
    # 数据清洗：修正 <= 0 的间隔
    intervals[intervals <= 0] = 1e-6
    return intervals

# ==========================================
# 1. 参数估计函数
# ==========================================

def jm_model_parameter_estimation(data, ex=1e-4, ey=1e-4):
    """
    JM模型参数估计
    注意：为了容错，这里先对数据进行预处理
    """
    # 自动识别并转换数据格式
    cumulative_data = preprocess_data(data)
    intervals = cumulative_to_intervals(cumulative_data)
    
    n = len(intervals)
    if n < 2: return None, None
        
    T = np.sum(intervals) # 总测试时间 = 最后一个累计时间
    S = 0.0
    for i in range(1, n + 1):
        S += (i - 1) * intervals[i-1]
        
    Q = S / T
    
    def func(N):
        sum_term = 0.0
        for i in range(1, n + 1):
            denom = N - i + 1
            if denom <= 0: return 1e9 # 避免无穷大报错
            sum_term += 1.0 / denom
        
        denom_right = N - Q
        if denom_right == 0: return 1e9
        
        return sum_term - n / denom_right

    # 扩大搜索范围
    left = n + 0.1
    right = n * 10000 
    
    f_left = func(left)
    
    # 动态寻找变号区间
    found_interval = False
    for _ in range(100):
        f_right = func(right)
        if f_left * f_right < 0:
            found_interval = True
            break
        right *= 2
        
    # 如果无解 (数据不符合 JM 模型假设)
    if not found_interval:
        # 降级处理
        N0_est = float(n) * 1.1 
        # 重新计算 phi
        denom = N0_est * T - S
        phi_est = n / denom if denom > 0 else 1e-4
        return N0_est, phi_est

    # 二分法求解
    mid = left
    for _ in range(100):
        mid = (left + right) / 2
        if abs(right - left) < ex:
            break
        f_mid = func(mid)
        if abs(f_mid) < ey:
            break
        if f_mid * f_left < 0:
            right = mid
        else:
            left = mid
            f_left = f_mid
            
    N0_est = mid
    denominator = N0_est * T - S
    phi_est = n / denominator if denominator > 0 else 1e-6
        
    return N0_est, phi_est

# ==========================================
# 2. 预测函数
# ==========================================

def jm_predict_future_failures(N0, phi, data, prediction_step):
    # 同样先处理数据，获取正确的当前状态
    cumulative_data = preprocess_data(data)
    n = len(cumulative_data)
    current_cumulative_time = cumulative_data[-1] if n > 0 else 0
    
    predicted_intervals = []
    cumulative_times = [] 
    
    # 修正剩余故障数逻辑
    remaining_faults = max(0, N0 - n)
    
    warning_msg = None
    # 如果 N0 极其巨大（例如 > 10^10），说明模型退化为常数失效率模型
    if N0 > 1e9:
        warning_msg = "警告：模型计算出 N0 极大，意味着数据没有表现出明显的可靠性增长。预测结果将接近平均间隔。"
    elif remaining_faults < 1:
        warning_msg = "警告：模型估算的剩余故障数已接近0，后续预测可能无效。"
    
    temp_cumulative = current_cumulative_time
    
    for k in range(1, prediction_step + 1):
        # 公式: MTTF = 1 / (phi * (N0 - (n + k - 1)))
        current_faults_found = n + k - 1
        denom = N0 - current_faults_found
        
        if denom <= 0:
            pred_interval = 0 
        else:
            pred_interval = 1.0 / (phi * denom)
            
        predicted_intervals.append(pred_interval)
        if pred_interval > 0:
            temp_cumulative += pred_interval
        cumulative_times.append(temp_cumulative)

    # 修正可靠度曲线计算: R(t|tn) = exp(-phi * (N0 - n) * t)
    future_duration = sum(predicted_intervals)
    if future_duration == 0 or future_duration > 1e6: # 防止绘图范围过大
        future_duration = np.mean(predicted_intervals) * 10 if predicted_intervals else 100
        
    t_points = np.linspace(0, future_duration, 50)
    curr_rem = max(0, N0 - n)
    reliability_values = [math.exp(-phi * curr_rem * t) for t in t_points]
        
    return {
        'remaining_faults': remaining_faults,
        'next_failure_time': cumulative_times[0] if cumulative_times else current_cumulative_time,
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
    # 确保使用正确的数据形式进行评估
    actual_cumulative = preprocess_data(data)
    n = len(actual_cumulative)
    
    if n == 0: return {'mae': 0, 'mse': 0, 'rmse': 0, 'r2_score': 0, 'accuracy': 0}

    predicted_cumulative = []
    current_val = 0.0
    
    # 重新模拟历史过程
    for i in range(1, n + 1):
        denom = N0 - (i - 1)
        interval = 1.0 / (phi * denom) if denom > 0 else 0
        current_val += interval
        predicted_cumulative.append(current_val)
        
    predicted_cumulative = np.array(predicted_cumulative)
    errors = actual_cumulative - predicted_cumulative
    
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual_cumulative - np.mean(actual_cumulative)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs(errors / actual_cumulative))
        accuracy = max(0, (1 - mape) * 100)
    
    return {
        'mae': mae, 'mse': mse, 'rmse': rmse, 'r2_score': r2, 'accuracy': accuracy
    }

# ==========================================
# 4. 辅助函数
# ==========================================

def calculate_reliability(N0, phi, times):
    times = np.array(times)
    return np.exp(-phi * N0 * times)

def plot_prediction_results(train_times, prediction_results, filename='prediction.png'):
    plt.figure(figsize=(10, 6))
    
    # 使用预处理后的数据绘图
    actual_cumulative = preprocess_data(train_times)
    n = len(actual_cumulative)
    plt.scatter(range(1, n + 1), actual_cumulative, color='blue', label='Actual Failures')
    
    pred_cumulative = prediction_results['cumulative_times']
    if pred_cumulative:
        start_idx = n + 1
        end_idx = n + len(pred_cumulative)
        plt.scatter(range(start_idx, end_idx + 1), pred_cumulative, color='red', marker='x', label='Predicted Failures')
    
    plt.xlabel('Failure Number')
    plt.ylabel('Cumulative Time')
    plt.title('JM Model Prediction')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()