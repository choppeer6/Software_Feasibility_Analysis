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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


def go_model_parameter_estimation(cumulative_times, method='mle', tolerance=1e-6):
    """
    GO模型参数估计 - 修正版本，直接处理累计失效时间

    参数:
        cumulative_times: 累计失效时间数组
        method: 估计方法，'mle'（最大似然估计）或 'ls'（最小二乘）
        tolerance: 收敛容差

    返回:
        a, b: 模型参数（最终故障数，故障检测率）
        若估计失败返回None, None

    异常:
        ValueError: 输入数据无效时抛出
    """
    # 输入验证
    if not isinstance(cumulative_times, (list, np.ndarray)):
        raise ValueError("cumulative_times参数必须是列表或numpy数组")

    cumulative_times = np.array(cumulative_times)

    if len(cumulative_times) < 2:
        raise ValueError("需要至少2个失效时间点才能进行参数估计")

    if np.any(cumulative_times <= 0):
        raise ValueError("失效时间必须为正数")

    n = len(cumulative_times)

    if method == 'mle':
        # 使用与GO.py完全一致的极大似然估计方法
        try:
            # 初始参数估计（与GO.py完全一致）
            a_init = n  # a的初始值：总失效数
            b_init = 1.0 / np.mean(cumulative_times)  # b的初始值：1/平均失效时间

            # 定义与GO.py完全一致的对数似然函数
            def negative_log_likelihood(params):
                a, b = params
                if a <= 0 or b <= 0:
                    return np.inf  # 返回很大的正值（因为是最小化负对数似然）

                # 计算对数似然（与GO.py完全一致）
                log_likelihood_val = 0.0
                for ti in cumulative_times:
                    # 失效强度函数 λ(t) = a * b * exp(-b * t)
                    intensity = a * b * np.exp(-b * ti)
                    if intensity <= 0:
                        return np.inf
                    log_likelihood_val += np.log(intensity)

                # 减去最终的累积失效数 m(t_n) = a * (1 - exp(-b * t_n))
                log_likelihood_val -= a * (1 - np.exp(-b * cumulative_times[-1]))

                # 返回负对数似然（用于最小化）
                return -log_likelihood_val

            # 参数边界 a>0, b>0（与GO.py一致）
            bounds = [(1e-6, None), (1e-6, None)]

            # 使用Nelder-Mead方法进行优化（与GO.py一致）
            result = minimize(
                negative_log_likelihood,
                [a_init, b_init],
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 10000, 'ftol': 1e-10}
            )

            if result.success:
                a_est, b_est = result.x
                # 确保参数有效并转换为Python标量
                if a_est > 0 and b_est > 0:
                    print(f"参数估计成功!")
                    print(f"估计的 a 值: {a_est:.4f}")
                    print(f"估计的 b 值: {b_est:.4f}")
                    return float(a_est), float(b_est)
                else:
                    raise ValueError("估计得到的参数无效")
            else:
                # 如果Nelder-Mead失败，尝试使用L-BFGS-B作为备选
                print("Nelder-Mead方法失败，尝试L-BFGS-B方法")
                result = minimize(
                    negative_log_likelihood,
                    [a_init, b_init],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': tolerance}
                )

                if result.success:
                    a_est, b_est = result.x
                    if a_est > 0 and b_est > 0:
                        print(f"参数估计成功 (使用L-BFGS-B)!")
                        print(f"估计的 a 值: {a_est:.4f}")
                        print(f"估计的 b 值: {b_est:.4f}")
                        return float(a_est), float(b_est)
                    else:
                        raise ValueError("估计得到的参数无效")
                else:
                    raise ValueError(f"参数优化失败: {result.message}")

        except Exception as e:
            # 如果MLE失败，尝试最小二乘法
            print(f"MLE方法失败，尝试最小二乘法: {str(e)}")
            method = 'ls'

    if method == 'ls':
        # 最小二乘法（备选方法）
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


def go_predict_future_failures(a, b, cumulative_times, num_predictions):
    """
    GO模型预测未来失效 - 直接使用累计失效时间

    参数:
        a, b: 模型参数
        cumulative_times: 累计失效时间数组
        num_predictions: 要预测的未来失效次数

    返回:
        dict: 包含各种预测结果
    """
    # 输入验证
    if not isinstance(cumulative_times, (list, np.ndarray)):
        raise ValueError("cumulative_times参数必须是列表或numpy数组")

    if num_predictions < 1:
        raise ValueError("预测步数必须至少为1")

    if a <= 0 or b <= 0:
        raise ValueError("模型参数a和b必须为正数")

    cumulative_times = np.array(cumulative_times)
    current_failures = len(cumulative_times)
    current_time = cumulative_times[-1] if len(cumulative_times) > 0 else 0

    # 计算当前累计失效数
    current_cumulative_failures = go_model_predict(a, b, current_time)

    # 计算剩余故障数 - 确保是标量
    if isinstance(current_cumulative_failures, np.ndarray):
        current_cumulative_failures = current_cumulative_failures.item() if current_cumulative_failures.size == 1 else \
        current_cumulative_failures[0]

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
            'reliability_curve': ([], []),
            'next_failure_time': None,
            'remaining_faults': max(0, float(remaining_faults))  # 确保是标量
        }

    # 预测未来的失效时间间隔（使用与GO.py一致的数值方法）
    predicted_intervals = []
    cumulative_times_pred = []
    current_pred_time = current_time

    for i in range(num_predictions):
        # 使用与GO.py一致的数值方法预测失效间隔
        target_increment = 1
        time_guess = current_pred_time + 1 / (a * b * np.exp(-b * current_pred_time))

        # 使用简单的迭代方法（与GO.py一致）
        t_test = time_guess
        for _ in range(100):
            increment = a * (1 - np.exp(-b * t_test)) - a * (1 - np.exp(-b * current_pred_time))
            if abs(increment - target_increment) < 0.01:
                break
            # 调整时间估计
            if increment < target_increment:
                t_test += 0.1
            else:
                t_test -= 0.1

        interval = t_test - current_pred_time
        predicted_intervals.append(float(interval))  # 确保是Python浮点数
        cumulative_times_pred.append(float(t_test))  # 确保是Python浮点数
        current_pred_time = t_test

    # 计算可靠度预测曲线（与GO.py一致）
    max_time = max(current_time * 1.5, 100) if current_time > 0 else 100
    time_points = np.linspace(0, max_time, 100)
    reliability_predictions = calculate_reliability(a, b, time_points)

    # 转换为Python列表
    time_points_list = time_points.tolist()
    reliability_predictions_list = reliability_predictions.tolist() if isinstance(reliability_predictions,
                                                                                  np.ndarray) else reliability_predictions

    result = {
        'predicted_intervals': predicted_intervals,
        'cumulative_times': cumulative_times_pred,
        'reliability_curve': (time_points_list, reliability_predictions_list),
        'next_failure_time': float(cumulative_times_pred[0]) if num_predictions > 0 else None,
        'remaining_faults': float(remaining_faults)  # 确保是Python浮点数
    }

    if warning_message:
        result['warning'] = warning_message

    return result


# def plot_prediction_results(cumulative_times, prediction_results):
#     """绘制预测结果并保存为图片"""
#     plt.figure(figsize=(12, 8))
#
#     # 绘制可靠度曲线
#     time_points, reliability = prediction_results['reliability_curve']
#     plt.subplot(2, 1, 1)
#     plt.plot(time_points, reliability, 'b-', label='Reliability Curve')
#     plt.axhline(y=0.5, color='r', linestyle='--', label='50% Reliability')
#
#     # 如果有警告信息，添加到图表中
#     if 'warning' in prediction_results:
#         plt.text(0.05, 0.05, prediction_results['warning'],
#                  transform=plt.gca().transAxes,
#                  color='red', fontweight='bold',
#                  bbox=dict(facecolor='yellow', alpha=0.5))
#
#     plt.xlabel('Time')
#     plt.ylabel('Reliability')
#     plt.title('System Reliability Over Time (GO Model)')
#     plt.grid(True)
#     plt.legend()
#
#     # 绘制失效时间预测
#     plt.subplot(2, 1, 2)
#     cumulative_times_pred = prediction_results['cumulative_times']
#
#     if len(cumulative_times) > 0:
#         # 历史失效时间
#         plt.scatter(range(1, len(cumulative_times) + 1), cumulative_times,
#                     color='g', label='Historical Failures', s=50)
#
#     if len(cumulative_times_pred) > 0:
#         # 预测的失效时间
#         plt.scatter(range(len(cumulative_times) + 1, len(cumulative_times) + 1 + len(cumulative_times_pred)),
#                     cumulative_times_pred, color='r', label='Predicted Failures', s=50)
#
#     plt.xlabel('Failure Sequence')
#     plt.ylabel('Time')
#     plt.title('Failure Time Prediction (GO Model)')
#     plt.grid(True)
#     plt.legend()
#
#     plt.tight_layout()
#     # 保存图片
#     plt.savefig('go_prediction_results.png', dpi=300, bbox_inches='tight')
#     print("预测结果已保存为 go_prediction_results.png")


def calculate_model_accuracy(a, b, cumulative_times):
    """
    计算模型预测准确率 - 与GO.py一致的算法

    参数:
        a, b: 模型参数
        cumulative_times: 累计失效时间数组

    返回:
        dict: 包含各种准确率指标
    """
    if len(cumulative_times) < 2:
        return {
            'mae': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'r2_score': 0.0,
            'accuracy': 0.0
        }

    # 计算模型预测的累计失效数
    predicted_cumulative_failures = go_model_predict(a, b, cumulative_times)

    # 实际累计失效数
    observed_cumulative_failures = np.arange(1, len(cumulative_times) + 1)

    # 计算误差指标（与GO.py一致，使用sklearn.metrics）
    mae = mean_absolute_error(observed_cumulative_failures, predicted_cumulative_failures)
    mse = mean_squared_error(observed_cumulative_failures, predicted_cumulative_failures)
    rmse = np.sqrt(mse)
    r2 = r2_score(observed_cumulative_failures, predicted_cumulative_failures)

    # 计算准确率（改进版：使用更合理的公式，避免MAE过大时准确率为0）
    # 使用相对误差的对称形式，限制单点最大误差为100%
    mean_obs = np.mean(observed_cumulative_failures)
    if mean_obs > 0:
        # 使用相对误差，但限制在合理范围内
        relative_errors = np.abs(observed_cumulative_failures - predicted_cumulative_failures) / (mean_obs + 1e-10)
        relative_errors = np.minimum(relative_errors, 1.0)  # 限制单点最大误差为100%
        mape = np.mean(relative_errors)
        accuracy = max(0.0, min(100.0, (1.0 - mape) * 100.0))
    else:
        accuracy = 0.0

    return {
        'mae': float(mae),  # 确保是Python浮点数
        'mse': float(mse),  # 确保是Python浮点数
        'rmse': float(rmse),  # 确保是Python浮点数
        'r2_score': float(r2),  # 确保是Python浮点数
        'accuracy': float(accuracy)  # 确保是Python浮点数
    }


def test_full_prediction():
    """测试完整的预测流程"""
    # 测试数据（累计失效时间，与GO.py一致）
    cumulative_times = [9, 21, 32, 36, 43, 45, 50, 58, 63, 70, 71, 77, 78, 87, 91, 92, 95, 103, 109, 110, 111, 144, 151,
                        242, 244, 245, 332, 379, 391, 400, 535, 793, 809, 844]

    try:
        # 拟合模型
        a, b = go_model_parameter_estimation(cumulative_times)
        print(f"拟合的GO模型参数: a={a:.4f}, b={b:.6f}")

        # 预测未来5次失效
        num_predictions = 5
        prediction_results = go_predict_future_failures(a, b, cumulative_times, num_predictions)

        # 打印预测结果 - 修复格式化错误
        print("\n预测结果:")
        print(f"剩余故障数: {float(prediction_results['remaining_faults']):.4f}")

        next_failure_time = prediction_results['next_failure_time']
        if next_failure_time is not None:
            print(f"下一次失效预测时间: {float(next_failure_time):.2f}")
        else:
            print("无剩余故障可预测")

        if 'warning' in prediction_results:
            print(f"\n警告: {prediction_results['warning']}")

        print("\n未来失效时间间隔预测:")
        for i, interval in enumerate(prediction_results['predicted_intervals']):
            print(f"第{i + 1}次预测失效间隔: {float(interval):.2f}")

        print("\n未来失效累积时间预测:")
        for i, time in enumerate(prediction_results['cumulative_times']):
            print(f"第{i + 1}次预测失效时间: {float(time):.2f}")

        # 计算模型准确率
        accuracy_metrics = calculate_model_accuracy(a, b, cumulative_times)
        print("\n模型准确率指标:")
        print(f"平均绝对误差 (MAE): {accuracy_metrics['mae']:.2f}")
        print(f"均方误差 (MSE): {accuracy_metrics['mse']:.2f}")
        print(f"均方根误差 (RMSE): {accuracy_metrics['rmse']:.2f}")
        print(f"决定系数 (R²): {accuracy_metrics['r2_score']:.4f}")
        print(f"准确率: {accuracy_metrics['accuracy']:.2f}%")

        # 绘制预测结果
        # plot_prediction_results(cumulative_times, prediction_results)

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_full_prediction()