# -*- coding: utf-8 -*-
"""
Goel-Okumoto (GO) 模型实现
用于软件可靠性分析和预测的非齐次泊松过程（NHPP）模型

模型特点：
- 累计失效函数：m(t) = a * (1 - e^(-b*t))
- 失效强度函数：λ(t) = a * b * e^(-b*t)
- 可靠度函数：R(t) = e^(-a*(1-e^(-b*t)))

其中：
- a: 最终故障数（asymptotic fault count）
- b: 故障检测率（fault detection rate）
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import io
from scipy.optimize import minimize
from scipy.optimize import fsolve

# 移除全局stdout重定向，避免与Flask冲突
# 如果需要处理编码问题，应该在具体函数中按需处理


def go_model_parameter_estimation(times, method='mle', tolerance=1e-6):
    """
    GO模型参数估计
    
    参数:
        times: 失效时间间隔数组（累计时间或时间间隔）
        method: 估计方法，'mle'（最大似然估计）或 'ls'（最小二乘）
        tolerance: 收敛容差
    
    返回:
        a, b: 模型参数（最终故障数，故障检测率）
        若估计失败返回None, None
    
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
        raise ValueError("失效时间必须为正数")
    
    # 转换为累计时间
    if len(times) > 1:
        cumulative_times = np.cumsum(times)
    else:
        cumulative_times = times
    
    n = len(cumulative_times)
    
    if method == 'mle':
        # 最大似然估计方法
        try:
            # 初始参数估计（使用简单启发式方法）
            # a的初始值：略大于观测到的故障数
            a_init = max(n * 1.2, n + 1)
            # b的初始值：基于平均故障间隔时间
            mean_time = np.mean(cumulative_times)
            b_init = 1.0 / mean_time if mean_time > 0 else 0.01
            
            # 定义负对数似然函数
            def negative_log_likelihood(params):
                a, b = params
                if a <= 0 or b <= 0:
                    return 1e10  # 返回一个很大的值
                
                # 计算对数似然
                log_likelihood = 0
                for i, t in enumerate(cumulative_times):
                    # 避免数值溢出
                    exp_term = np.exp(-b * t)
                    if exp_term <= 0 or exp_term >= 1:
                        return 1e10
                    
                    # 累计失效函数
                    m_t = a * (1 - exp_term)
                    
                    # 失效强度函数
                    lambda_t = a * b * exp_term
                    
                    if lambda_t <= 0:
                        return 1e10
                    
                    # 对数似然贡献
                    if i == 0:
                        # 第一个失效
                        log_likelihood += np.log(lambda_t) - m_t
                    else:
                        # 后续失效
                        prev_t = cumulative_times[i-1]
                        prev_m = a * (1 - np.exp(-b * prev_t))
                        log_likelihood += np.log(lambda_t) - (m_t - prev_m)
                
                return -log_likelihood  # 返回负值用于最小化
            
            # 使用优化算法估计参数
            result = minimize(
                negative_log_likelihood,
                [a_init, b_init],
                method='L-BFGS-B',
                bounds=[(n, n * 10), (1e-6, 10)],
                options={'maxiter': 1000, 'ftol': tolerance}
            )
            
            if result.success:
                a, b = result.x
                # 确保参数有效并转换为Python标量
                if a > 0 and b > 0:
                    return float(a), float(b)
                else:
                    raise ValueError("估计得到的参数无效")
            else:
                raise ValueError(f"参数优化失败: {result.message}")
        
        except Exception as e:
            # 如果MLE失败，尝试最小二乘法
            print(f"MLE方法失败，尝试最小二乘法: {str(e)}")
            method = 'ls'
    
    if method == 'ls':
        # 最小二乘法
        try:
            # 使用非线性最小二乘拟合
            def residuals(params):
                a, b = params
                if a <= 0 or b <= 0:
                    return np.ones(n) * 1e10
                
                predicted = []
                for t in cumulative_times:
                    m_t = a * (1 - np.exp(-b * t))
                    predicted.append(m_t)
                
                predicted = np.array(predicted)
                observed = np.arange(1, n + 1)  # 累计失效数
                
                return predicted - observed
            
            # 初始参数
            a_init = max(n * 1.2, n + 1)
            mean_time = np.mean(cumulative_times)
            b_init = 1.0 / mean_time if mean_time > 0 else 0.01
            
            # 求解
            result = minimize(
                lambda params: np.sum(residuals(params) ** 2),
                [a_init, b_init],
                method='L-BFGS-B',
                bounds=[(n, n * 10), (1e-6, 10)],
                options={'maxiter': 1000, 'ftol': tolerance}
            )
            
            if result.success:
                a, b = result.x
                if a > 0 and b > 0:
                    return float(a), float(b)
                else:
                    raise ValueError("估计得到的参数无效")
            else:
                raise ValueError(f"最小二乘估计失败: {result.message}")
        
        except Exception as e:
            raise ValueError(f"参数估计失败: {str(e)}")
    
    return None, None


def go_model_predict(a, b, times):
    """
    使用GO模型参数进行预测
    
    参数:
        a: 最终故障数
        b: 故障检测率
        times: 时间点数组
    
    返回:
        预测的累计失效数数组
    """
    # 输入验证
    if not isinstance(times, (list, np.ndarray)):
        times = np.array([times])
    else:
        times = np.array(times)
    
    if np.any(times < 0):
        raise ValueError("时间点不能为负数")
    
    if a <= 0 or b <= 0:
        raise ValueError("模型参数a和b必须为正数")
    
    # GO模型的累计失效函数
    predictions = a * (1 - np.exp(-b * times))
    
    return predictions


def calculate_reliability(a, b, times):
    """
    计算系统在给定时间点的可靠度
    
    参数:
        a: 最终故障数
        b: 故障检测率
        times: 时间点数组
    
    返回:
        可靠度数组
    """
    # GO模型的可靠度函数：R(t) = e^(-a*(1-e^(-b*t)))
    reliability = np.exp(-a * (1 - np.exp(-b * times)))
    return reliability


def go_predict_future_failures(a, b, train_times, num_predictions):
    """
    GO模型预测未来失效
    
    参数:
        a, b: 模型参数
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
    
    if a <= 0 or b <= 0:
        raise ValueError("模型参数a和b必须为正数")
    
    train_times = np.array(train_times)
    current_failures = len(train_times)
    current_time = np.sum(train_times)
    
    # 计算当前累计失效数
    current_cumulative_failures = go_model_predict(a, b, current_time)
    
    # 计算剩余故障数
    remaining_faults = a - current_cumulative_failures
    
    if remaining_faults <= 0:
        warning_message = f"警告：最终故障数a({a:.4f})小于等于当前累计失效数({current_cumulative_failures:.4f})，预测结果可能不准确"
        print(warning_message)
        
        # 限制预测步数
        max_possible_predictions = 3
        num_predictions = min(num_predictions, max_possible_predictions)
        
        # 使用调整后的剩余故障数
        effective_remaining_faults = max(0.1 * a, 1)
    else:
        warning_message = None
        effective_remaining_faults = remaining_faults
    
    # 限制预测步数不超过剩余故障数
    max_possible_predictions = max(0, int(np.ceil(effective_remaining_faults)))
    num_predictions = min(num_predictions, max_possible_predictions)
    
    if num_predictions == 0:
        return {
            'predicted_intervals': [],
            'cumulative_times': [],
            'reliability_curve': (np.array([]), np.array([])),
            'next_failure_time': None,
            'remaining_faults': max(0, remaining_faults)
        }
    
    # 预测未来的失效时间间隔
    predicted_intervals = []
    cumulative_times = []
    current_pred_time = current_time
    
    for i in range(num_predictions):
        # 计算下一次失效的预期时间
        # 使用失效强度函数：λ(t) = a * b * e^(-b*t)
        # 对于下一次失效，我们需要求解：∫[t_current, t_next] λ(τ) dτ = 1
        
        # 简化方法：使用当前失效强度估算时间间隔
        current_lambda = a * b * np.exp(-b * current_pred_time)
        
        if current_lambda > 0:
            # 时间间隔 = 1 / 失效强度
            interval = 1.0 / current_lambda
        else:
            # 如果失效强度太小，使用较大的间隔
            interval = 100.0
        
        predicted_intervals.append(interval)
        current_pred_time += interval
        cumulative_times.append(current_pred_time)
    
    predicted_intervals = np.array(predicted_intervals)
    cumulative_times = np.array(cumulative_times)
    
    # 计算可靠度预测曲线
    # 从时间0开始，包含历史数据和预测数据，以便显示完整的可靠性变化
    max_time = current_time + np.sum(predicted_intervals)
    # 使用更多的时间点，从0开始到预测结束时间
    time_points = np.linspace(0, max_time, 200)
    reliability_predictions = calculate_reliability(a, b, time_points)
    
    result = {
        'predicted_intervals': predicted_intervals,
        'cumulative_times': cumulative_times,
        'reliability_curve': (time_points, reliability_predictions),
        'next_failure_time': cumulative_times[0] if num_predictions > 0 else None,
        'remaining_faults': remaining_faults
    }
    
    if warning_message:
        result['warning'] = warning_message
    
    return result


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
    plt.title('System Reliability Over Time (GO Model)')
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
    plt.title('Failure Time Prediction (GO Model)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    # 保存图片
    plt.savefig('go_prediction_results.png', dpi=300, bbox_inches='tight')
    print("预测结果已保存为 go_prediction_results.png")


def calculate_model_accuracy(a, b, actual_times):
    """
    计算模型预测准确率
    
    参数:
        a, b: 模型参数
        actual_times: 实际失效时间间隔数组
    
    返回:
        dict: 包含各种准确率指标
    """
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
    
    # 计算模型预测的累计失效数
    predicted_cumulative_failures = go_model_predict(a, b, actual_cumulative)
    
    # 实际累计失效数
    observed_cumulative_failures = np.arange(1, len(actual_cumulative) + 1)
    
    # 计算误差指标
    mae = np.mean(np.abs(observed_cumulative_failures - predicted_cumulative_failures))
    mse = np.mean((observed_cumulative_failures - predicted_cumulative_failures) ** 2)
    rmse = np.sqrt(mse)
    
    # 计算R²分数
    ss_total = np.sum((observed_cumulative_failures - np.mean(observed_cumulative_failures)) ** 2)
    ss_residual = np.sum((observed_cumulative_failures - predicted_cumulative_failures) ** 2)
    r2_score = 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0
    
    # 计算准确率（基于相对误差）
    relative_errors = np.abs((observed_cumulative_failures - predicted_cumulative_failures) / 
                             np.maximum(observed_cumulative_failures, 1))
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
    # 测试数据
    train_times = [9, 21, 32, 36, 43, 45, 50, 58, 63, 70, 71, 77, 78, 87, 91, 92, 95, 103, 109, 110, 111, 144, 151, 242, 244, 245, 332, 379, 391, 400, 535, 793, 809, 844]
    
    try:
        # 拟合模型
        a, b = go_model_parameter_estimation(train_times)
        print(f"拟合的GO模型参数: a={a:.4f}, b={b:.6f}")
        
        # 预测未来5次失效
        num_predictions = 10
        prediction_results = go_predict_future_failures(a, b, train_times, num_predictions)
        
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
        accuracy_metrics = calculate_model_accuracy(a, b, train_times)
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
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_full_prediction()

