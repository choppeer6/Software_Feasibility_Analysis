import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt
import warnings


def nhpp_model_parameter_estimation(failure_data, model_type='exponential'):
    """
    NHPP模型参数估计（使用最小二乘拟合累计失效曲线，数值更稳定）

    参数:
    - failure_data: 失效时间数据列表
    - model_type: 模型类型 ('exponential', 'power', 'logarithmic')

    返回:
    - params: 模型参数
    - model_type: 使用的模型类型
    """
    if len(failure_data) < 2:
        raise ValueError("至少需要2个失效数据点")

    failure_data = np.asarray(failure_data, dtype=float)
    n = len(failure_data)
    t_total = failure_data[-1]

    # 观测的累计失效数 1,2,...,n
    m_obs = np.arange(1, n + 1, dtype=float)

    if t_total <= 0:
        # 极端情况：时间轴不增长，退化为简单计数
        return np.array([float(n), 1.0]), model_type

    if model_type == 'exponential':
        # 指数型 NHPP: m(t) = a (1 - e^{-b t})
        def loss(params):
            a, b = params
            if a <= 0 or b <= 0:
                return 1e10
            m_hat = a * (1.0 - np.exp(-b * failure_data))
            return np.mean((m_hat - m_obs) ** 2)

        a0 = float(n * 1.5)
        b0 = float(1.0 / max(t_total, 1.0))
        bounds = [
            (max(n * 0.8, 1.0), n * 10.0),   # a 大致在已观测失效数附近
            (1e-6, 10.0 / t_total)          # b 控制衰减速度
        ]

        result = minimize(loss, [a0, b0], bounds=bounds)
        if result.success:
            return result.x, model_type
        return np.array([a0, b0]), model_type

    elif model_type == 'power':
        # 幂律型 NHPP: m(t) = a t^b
        def loss(params):
            a, b = params
            if a <= 0 or b <= 0:
                return 1e10
            m_hat = a * (failure_data ** b)
            return np.mean((m_hat - m_obs) ** 2)

        a0 = float(n / max(t_total ** 0.5, 1.0))
        b0 = 0.5
        bounds = [
            (1e-6, n * 10.0),
            (0.1, 5.0)
        ]

        result = minimize(loss, [a0, b0], bounds=bounds)
        if result.success:
            return result.x, model_type
        return np.array([a0, b0]), model_type

    else:  # logarithmic
        # 对数型 NHPP: m(t) = a \ln(1 + b t)
        def loss(params):
            a, b = params
            if a <= 0 or b <= 0:
                return 1e10
            m_hat = a * np.log(1.0 + b * failure_data)
            return np.mean((m_hat - m_obs) ** 2)

        a0 = float(n / max(np.log(1.0 + t_total), 1.0))
        b0 = float(1.0 / max(t_total, 1.0))
        bounds = [
            (1e-6, n * 10.0),
            (1e-6, 10.0 / t_total)
        ]

        result = minimize(loss, [a0, b0], bounds=bounds)
        if result.success:
            return result.x, model_type
        return np.array([a0, b0]), model_type


def nhpp_intensity_function(t, params, model_type):
    """强度函数"""
    if model_type == 'exponential':
        a, b = params
        return a * b * np.exp(-b * t)
    elif model_type == 'power':
        a, b = params
        return a * b * (t ** (b - 1))
    else:  # logarithmic
        a, b = params
        return (a * b) / (1 + b * t)


def nhpp_mean_function(t, params, model_type):
    """均值函数"""
    if model_type == 'exponential':
        a, b = params
        return a * (1 - np.exp(-b * t))
    elif model_type == 'power':
        a, b = params
        return a * (t ** b)
    else:  # logarithmic
        a, b = params
        return a * np.log(1 + b * t)


def nhpp_predict_future_failures(params, model_type, failure_data, prediction_step=5):
    """
    NHPP模型预测未来失效

    参数:
    - params: 模型参数
    - model_type: 模型类型
    - failure_data: 历史失效数据
    - prediction_step: 预测步长

    返回:
    - 预测结果字典
    """
    n = len(failure_data)
    t_current = failure_data[-1]

    # 计算当前累计失效数
    current_failures = n

    # 估计总故障数
    if model_type == 'exponential':
        a, b = params
        total_faults = a
    elif model_type == 'power':
        a, b = params
        # 对于幂律模型，总故障数需要设定一个时间上限
        time_horizon = t_current * 2  # 假设时间范围是当前时间的2倍
        total_faults = nhpp_mean_function(time_horizon, params, model_type)
    else:  # logarithmic
        a, b = params
        time_horizon = t_current * 2
        total_faults = nhpp_mean_function(time_horizon, params, model_type)

    remaining_faults = max(0, total_faults - current_failures)

    # 预测未来失效时间间隔：直接求解 m(t) = 当前累计失效数 + k
    predicted_intervals = []
    cumulative_times = []
    current_time = t_current
    current_count = current_failures

    def solve_time_for_count(target_count):
        """二分求解 m(t)=target_count."""
        # 上界逐步扩展直到均值超过目标
        lower = max(current_time, 0.0)
        upper = lower + max(10.0, lower * 0.2 + 10.0)
        mean_upper = nhpp_mean_function(upper, params, model_type)
        expand_steps = 0
        while mean_upper < target_count and expand_steps < 40:
            upper += max(10.0, upper * 0.5 + 10.0)
            mean_upper = nhpp_mean_function(upper, params, model_type)
            expand_steps += 1

        if mean_upper < target_count:
            return None

        for _ in range(60):
            mid = 0.5 * (lower + upper)
            mean_mid = nhpp_mean_function(mid, params, model_type)
            if mean_mid < target_count:
                lower = mid
            else:
                upper = mid
        return 0.5 * (lower + upper)

    for _ in range(prediction_step):
        target = current_count + 1

        # 如果模型预计的总故障数不足以支持更多失效，提前退出
        if total_faults <= target - 1:
            break

        solved_time = solve_time_for_count(target)
        if solved_time is None:
            break

        interval = max(solved_time - current_time, 0.0)
        predicted_intervals.append(interval)
        current_time = solved_time
        cumulative_times.append(current_time)
        current_count = target

    # 下一次失效时间
    next_failure_time = cumulative_times[0] if cumulative_times else None

    # 生成可靠度曲线数据
    horizon = max(t_current + sum(predicted_intervals), t_current * 1.5, 100)
    time_points = np.linspace(0, horizon, 120)
    reliability_values = []

    for t in time_points:
        # 可靠度 = P(在时间t内无失效)
        mean_val = nhpp_mean_function(t, params, model_type)
        reliability = np.exp(-mean_val)
        reliability_values.append(reliability)

    # 检查模型合理性
    warning = None
    if remaining_faults < 0:
        warning = "警告：估计的总故障数小于已观测故障数，模型可能需要调整"
    elif remaining_faults > current_failures * 10:
        warning = "警告：估计的剩余故障数异常高，模型可能需要调整"

    return {
        'remaining_faults': remaining_faults,
        'next_failure_time': next_failure_time,
        'predicted_intervals': predicted_intervals,
        'cumulative_times': cumulative_times,
        'reliability_curve': (time_points, reliability_values),
        'warning': warning,
        'total_faults': total_faults
    }


def calculate_nhpp_model_accuracy(params, model_type, failure_data):
    """
    计算NHPP模型准确率

    参数:
    - params: 模型参数
    - model_type: 模型类型
    - failure_data: 失效数据

    返回:
    - 准确率指标字典
    """
    n = len(failure_data)
    if n < 2:
        return {'mae': 0, 'mse': 0, 'rmse': 0, 'r2_score': 0, 'accuracy': 0}

    # 计算预测的累计失效数
    predicted_cumulative = []
    actual_cumulative = list(range(1, n + 1))

    for i, t in enumerate(failure_data):
        mean_val = nhpp_mean_function(t, params, model_type)
        predicted_cumulative.append(mean_val)

    predicted_cumulative = np.array(predicted_cumulative)
    actual_cumulative = np.array(actual_cumulative)

    # 计算误差指标
    mae = np.mean(np.abs(predicted_cumulative - actual_cumulative))
    mse = np.mean((predicted_cumulative - actual_cumulative) ** 2)
    rmse = np.sqrt(mse)

    # R² score
    ss_res = np.sum((actual_cumulative - predicted_cumulative) ** 2)
    ss_tot = np.sum((actual_cumulative - np.mean(actual_cumulative)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # 准确率（基于相对误差）
    relative_errors = np.abs((predicted_cumulative - actual_cumulative) / (actual_cumulative + 1e-10))
    accuracy = max(0, 100 * (1 - np.mean(relative_errors)))

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2_score,
        'accuracy': accuracy
    }


def plot_nhpp_prediction_results(params, model_type, failure_data, prediction_results, save_path=None):
    """
    绘制NHPP模型预测结果

    参数:
    - params: 模型参数
    - model_type: 模型类型
    - failure_data: 失效数据
    - prediction_results: 预测结果
    - save_path: 图片保存路径
    """
    plt.figure(figsize=(12, 8))

    # 子图1: 累计失效数
    plt.subplot(2, 2, 1)
    t_observed = np.array(failure_data)
    n_observed = len(t_observed)
    m_observed = np.arange(1, n_observed + 1)

    # 绘制观测数据
    plt.plot(t_observed, m_observed, 'bo-', label='观测数据', markersize=4)

    # 绘制模型拟合曲线
    t_model = np.linspace(0, max(t_observed) * 1.2, 100)
    m_model = nhpp_mean_function(t_model, params, model_type)
    plt.plot(t_model, m_model, 'r-', label='NHPP模型拟合')

    plt.xlabel('时间')
    plt.ylabel('累计失效数')
    plt.title('NHPP模型拟合')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: 强度函数
    plt.subplot(2, 2, 2)
    intensity = nhpp_intensity_function(t_model, params, model_type)
    plt.plot(t_model, intensity, 'g-', label='强度函数')
    plt.xlabel('时间')
    plt.ylabel('失效强度')
    plt.title('失效强度函数')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图3: 可靠度函数
    plt.subplot(2, 2, 3)
    time_points, reliability_values = prediction_results['reliability_curve']
    plt.plot(time_points, reliability_values, 'purple', label='可靠度')
    plt.xlabel('时间')
    plt.ylabel('可靠度')
    plt.title('系统可靠度')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图4: 预测结果
    plt.subplot(2, 2, 4)
    # 历史数据
    plt.plot(t_observed, m_observed, 'bo-', label='历史数据', markersize=4)

    # 预测数据
    if prediction_results['cumulative_times']:
        t_pred = prediction_results['cumulative_times']
        m_pred = list(range(n_observed + 1, n_observed + len(t_pred) + 1))
        plt.plot(t_pred, m_pred, 'ro--', label='预测数据', markersize=4)

    plt.xlabel('时间')
    plt.ylabel('累计失效数')
    plt.title('失效预测')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
