import numpy as np
import matplotlib.pyplot as plt
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
def jm_model_parameter_estimation(times, e_x=1e-5, e_y=1e-5):
    """JM模型参数估计
    
    参数:
        times: 失效时间间隔数组
        e_x: x方向容差
        e_y: y方向容差
    
    返回:
        N0, phi: 模型参数，若估计失败返回None, None
    
    异常:
        ValueError: 输入数据无效时抛出
    """
    # 输入验证
    if not isinstance(times, (list, np.ndarray)):
        raise ValueError("times参数必须是列表或numpy数组")
        
    times = np.array(times)
    
    if len(times) < 2:
        raise ValueError("需要至少2个失效时间点才能进行参数估计")
        
    if np.any(times <= 0):
        raise ValueError("失效时间间隔必须为正数")
    
    n = len(times)
    p = np.sum([(i-1) * times[i-1] for i in range(1, n+1)]) / np.sum(times)
    
    if p <= (n-1)/2:
        raise ValueError("无法估计模型参数，p值不符合要求")
    
    left = n - 1
    right = n
    
    def f(x):
        total = 0
        for i in range(1, n+1):
            if x <= (i-1):
                return float('inf')  # 避免除以零
            total += 1 / (x - (i-1))
        if x <= p:
            return float('-inf')  # 避免除以零
        return total - n / (x - p)
    
    # 查找合适的右边界
    while f(right) < -e_y:
        right += 1
    
    # 二分法求解
    max_iterations = 10000
    iteration = 0
    root = None
    
    while abs(right - left) > e_x and iteration < max_iterations:
        mid = (left + right) / 2
        f_mid = f(mid)
        
        if abs(f_mid) <= e_y:
            root = mid
            break
        elif f_mid > e_y:
            left = mid
        else:
            right = mid
        
        iteration += 1
    
    # 如果循环结束还没找到精确解，使用最后一次的中点
    if root is None:
        # 无论是到达最大迭代还是精度满足停止条件，都使用中点作为近似根
        root = (left + right) / 2
    
    N0 = root
    phi = n / (N0 * np.sum(times) - np.sum([(i-1) * times[i-1] for i in range(1, n+1)]))
    
    if phi <= 0:
        raise ValueError("估计得到的phi参数无效（<=0）")
    
    # 确保N0足够大，避免后续预测时出现问题
    if N0 < n:
        # 如果N0略小于n，稍微调整一下
        adjustment_factor = 1.05  # 增加5%
        N0 = max(N0 * adjustment_factor, n + 1e-6)
        # 重新计算phi
        phi = n / (N0 * np.sum(times) - np.sum([(i-1) * times[i-1] for i in range(1, n+1)]))
    
    return N0, phi

def jm_model_predict(N0, phi, times):
    """使用JM模型参数进行预测
    
    参数:
        N0: 初始故障数
        phi: 故障检测率
        times: 时间点数组
    
    返回:
        预测的累计失效概率数组
    """
    # 输入验证
    if not isinstance(times, (list, np.ndarray)):
        times = np.array([times])
    else:
        times = np.array(times)
    
    if np.any(times < 0):
        raise ValueError("时间点不能为负数")
        
    if N0 <= 0 or phi <= 0:
        raise ValueError("模型参数N0和phi必须为正数")
    
    # 使用向量化运算提升效率
    predictions = np.zeros_like(times, dtype=np.float64)
    
    # 处理t <= 0的情况
    mask = times > 0
    predictions[mask] = 1 - np.exp(-phi * (times[mask] - 1))
    
    return predictions

def calculate_reliability(N0, phi, times):
    """计算系统在给定时间点的可靠度"""
    probabilities = jm_model_predict(N0, phi, times)
    return 1 - probabilities

def jm_predict_future_failures(N0, phi, train_times, num_predictions):
    """
    JM模型预测未来失效
    
    参数:
        N0, phi: 模型参数
        train_times: 训练数据(历史失效时间间隔)
        num_predictions: 要预测的未来失效次数
        
    返回:
        dict: 包含各种预测结果
    """
    # 输入验证
    if not isinstance(train_times, (list, np.ndarray)):
        raise ValueError("train_times参数必须是列表或numpy数组")
        
    if num_predictions < 1:
        raise ValueError("预测步数必须至少为1")
        
    if N0 <= 0 or phi <= 0:
        raise ValueError("模型参数N0和phi必须为正数")
    
    train_times = np.array(train_times)
    current_failures = len(train_times)
    current_time = np.sum(train_times)
    
    remaining_faults = N0 - current_failures
    
    # 处理边界情况，允许N0略小于current_failures
    if remaining_faults <= 0:
        # 如果剩余故障数为0或负数，仍然尝试预测几次
        warning_message = f"警告：初始故障数N0({N0:.4f})小于等于已发生的故障数({current_failures})，预测结果可能不准确"
        print(warning_message)

        # 限制预测步数
        max_possible_predictions = 3  # 即使没有剩余故障，也尝试预测3次
        num_predictions = min(num_predictions, max_possible_predictions)

        # 使用当前故障数的90%作为剩余故障数的下限（修正原始表达式的符号错误）
        effective_remaining_faults = max(0.9 * current_failures, 1)

        # 预测未来的失效时间间隔（向量化实现）
        remaining_faults_sequence = effective_remaining_faults - np.arange(num_predictions)
        # 确保不出现负数
        remaining_faults_sequence = np.maximum(remaining_faults_sequence, 1e-6)

        predicted_intervals = 1 / (phi * remaining_faults_sequence)

        # 计算累积时间预测
        cumulative_times = current_time + np.cumsum(predicted_intervals)

        # 计算可靠度预测
        time_points = np.linspace(0, np.sum(predicted_intervals), 100)
        reliability_predictions = np.exp(-phi * effective_remaining_faults * time_points)

        return {
            'predicted_intervals': predicted_intervals,
            'cumulative_times': cumulative_times,
            'reliability_curve': (time_points, reliability_predictions),
            'next_failure_time': cumulative_times[0] if num_predictions > 0 else None,
            'remaining_faults': remaining_faults,
            'warning': warning_message
        }
    
    # 确保不会预测超过剩余故障数的失效
    import math
    # 使用向上取整，允许在剩余故障为小数时多预测一条（更乐观的策略）
    max_possible_predictions = max(0, int(math.ceil(remaining_faults)))
    num_predictions = min(num_predictions, max_possible_predictions)
    
    if num_predictions == 0:
        return {
            'predicted_intervals': [],
            'cumulative_times': [],
            'reliability_curve': (np.array([]), np.array([])),
            'next_failure_time': None,
            'remaining_faults': max(0, remaining_faults)
        }
    
    # 预测未来的失效时间间隔（向量化实现）
    remaining_faults_sequence = remaining_faults - np.arange(num_predictions)
    predicted_intervals = 1 / (phi * remaining_faults_sequence)
    
    # 计算累积时间预测
    cumulative_times = current_time + np.cumsum(predicted_intervals)
    
    # 计算可靠度预测
    time_points = np.linspace(0, np.sum(predicted_intervals), 100)
    reliability_predictions = np.exp(-phi * remaining_faults * time_points)
    
    return {
        'predicted_intervals': predicted_intervals,
        'cumulative_times': cumulative_times,
        'reliability_curve': (time_points, reliability_predictions),
        'next_failure_time': cumulative_times[0] if num_predictions > 0 else None,
        'remaining_faults': remaining_faults
    }

def plot_prediction_results(train_times, prediction_results):
    """绘制预测结果并保存为图片"""
    plt.figure(figsize=(12, 8))
    
    # 绘制可靠度曲线
    time_points, reliability = prediction_results['reliability_curve']
    plt.subplot(2, 1, 1)
    plt.plot(time_points, reliability, 'b-', label='Reliability Curve')
    plt.axhline(y=0.5, color='r', linestyle='--', label='50% Reliability')
    
    # 如果有警告信息，添加到图表中
    if 'warning' in prediction_results:
        plt.text(0.05, 0.05, prediction_results['warning'], 
                transform=plt.gca().transAxes, 
                color='red', fontweight='bold',
                bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.xlabel('Time')
    plt.ylabel('Reliability')
    plt.title('System Reliability Over Time')
    plt.grid(True)
    plt.legend()
    
    # 绘制失效时间预测
    plt.subplot(2, 1, 2)
    cumulative_times = prediction_results['cumulative_times']
    
    if len(train_times) > 0:
        # 历史失效时间
        historical_times = np.cumsum(train_times)
        plt.scatter(range(1, len(historical_times)+1), historical_times, 
                   color='g', label='Historical Failures', s=50)
    
    if len(cumulative_times) > 0:
        # 预测的失效时间
        plt.scatter(range(len(historical_times)+1, len(historical_times)+1+len(cumulative_times)), 
                   cumulative_times, color='r', label='Predicted Failures', s=50)
    
    plt.xlabel('Failure Sequence')
    plt.ylabel('Time')
    plt.title('Failure Time Prediction')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    # 保存图片而不是显示
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    print("预测结果已保存为 prediction_results.png")

def calculate_model_accuracy(N0, phi, actual_times):
    """计算模型预测准确率"""
    if len(actual_times) < 2:
        return {
            'mae': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'r2_score': 0.0,
            'accuracy': 0.0
        }
    
    # 计算实际累积失效时间
    actual_cumulative = np.cumsum(actual_times)
    
    # 计算模型预测的累积失效时间
    predicted_cumulative = np.zeros_like(actual_cumulative)
    
    for i in range(len(actual_cumulative)):
        t = actual_cumulative[i]
        remaining_faults = N0 - i
        if remaining_faults <= 0:
            predicted_cumulative[i] = float('inf')
        else:
            predicted_cumulative[i] = (1/phi) * np.log(N0 / remaining_faults)
    
    # 计算误差指标
    mask = predicted_cumulative != float('inf')
    if not np.any(mask):
        return {
            'mae': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'r2_score': 0.0,
            'accuracy': 0.0
        }
    
    actual = actual_cumulative[mask]
    predicted = predicted_cumulative[mask]
    
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    
    # 计算R²分数
    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    ss_residual = np.sum((actual - predicted) ** 2)
    r2_score = 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0
    
    # 计算准确率（基于相对误差）
    relative_errors = np.abs((actual - predicted) / actual)
    accuracy = np.mean(1 - relative_errors) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2_score,
        'accuracy': accuracy
    }

def test_full_prediction():
    """测试完整的预测流程"""
    # 真实数据
    train_times = [9, 21, 32, 36, 43, 45, 50, 58, 63, 70, 71, 77, 78, 87, 91, 92, 95, 103, 109, 110, 111, 144, 151, 242, 244, 245, 332, 379, 391, 400, 535, 793, 809, 844]
    
    try:
        # 拟合模型
        N0, phi = jm_model_parameter_estimation(train_times)
        print(f"拟合的JM模型参数: N0={N0:.4f}, φ={phi:.4f}")
        
        # 预测未来5次失效
        num_predictions = 20
        prediction_results = jm_predict_future_failures(N0, phi, train_times, num_predictions)
        
        # 打印预测结果
        print("\n预测结果:")
        print(f"剩余故障数: {prediction_results['remaining_faults']:.4f}")
        print(f"下一次失效预测时间: {prediction_results['next_failure_time']:.2f}" 
              if prediction_results['next_failure_time'] else "无剩余故障可预测")
        
        if 'warning' in prediction_results:
            print(f"\n警告: {prediction_results['warning']}")
        
        print("\n未来失效时间间隔预测:")
        for i, interval in enumerate(prediction_results['predicted_intervals']):
            print(f"第{i+1}次预测失效间隔: {interval:.2f}")
        
        print("\n未来失效累积时间预测:")
        for i, time in enumerate(prediction_results['cumulative_times']):
            print(f"第{i+1}次预测失效时间: {time:.2f}")
        
        # 计算模型准确率
        accuracy_metrics = calculate_model_accuracy(N0, phi, train_times)
        print("\n模型准确率指标:")
        print(f"平均绝对误差 (MAE): {accuracy_metrics['mae']:.2f}")
        print(f"均方误差 (MSE): {accuracy_metrics['mse']:.2f}")
        print(f"均方根误差 (RMSE): {accuracy_metrics['rmse']:.2f}")
        print(f"决定系数 (R²): {accuracy_metrics['r2_score']:.4f}")
        print(f"准确率: {accuracy_metrics['accuracy']:.2f}%")
        
        # 绘制预测结果
        plot_prediction_results(train_times, prediction_results)
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    test_full_prediction()