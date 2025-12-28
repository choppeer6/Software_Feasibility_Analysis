import numpy as np
from typing import List, Dict, Union, Tuple


def _to_series(failure_data: Union[List[float], np.ndarray]) -> np.ndarray:
    """将失效时间间隔列表转为 numpy 序列，不排序以保持时间顺序。"""
    arr = np.asarray(failure_data, dtype=float)
    return arr


def gm11_train_model(failure_intervals: Union[List[float], np.ndarray]) -> Dict[str, Union[float, np.ndarray]]:
    """
    训练一阶灰色 GM(1,1) 模型，输入为失效时间间隔序列。

    模型形式：
        原始序列 X⁽⁰⁾ = {x⁽⁰⁾(1), x⁽⁰⁾(2), ..., x⁽⁰⁾(n)}
        累加序列 X⁽¹⁾(k) = Σ{i=1..k} x⁽⁰⁾(i)
        白化方程 dx⁽¹⁾/dt + θx⁽¹⁾ = u
        解：x̂⁽¹⁾(k+1) = (x⁽⁰⁾(1) - u/θ)e^{-θk} + u/θ
        还原值：x̂⁽⁰⁾(k+1) = (1 - e^{θ})(x⁽⁰⁾(1) - u/θ)e^{-θk}
    """
    x0 = _to_series(failure_intervals)  # 原始序列 X⁽⁰⁾
    n = x0.size
    if n < 4:
        raise ValueError("GM(1,1) 建议至少提供 4 个以上数据点")

    # 检查数据非负性
    if np.any(x0 < 0):
        raise ValueError("失效时间间隔必须为非负值")

    # 1-AGO 累加生成序列 X⁽¹⁾
    x1 = np.cumsum(x0)

    # 构造矩阵B和向量y_N (根据公式2-6和2-7)
    # B矩阵第一列：-0.5*(x⁽¹⁾(k) + x⁽¹⁾(k-1))
    B = np.ones((n - 1, 2))
    for i in range(1, n):
        B[i - 1, 0] = -0.5 * (x1[i] + x1[i - 1])

    y_N = x0[1:].reshape(-1, 1)

    # 最小二乘估计灰参数 [θ, u]^T
    BTB = B.T @ B
    if np.linalg.det(BTB) == 0:
        raise ValueError("GM(1,1) 参数矩阵奇异，无法拟合")
    a_hat = np.linalg.inv(BTB) @ B.T @ y_N
    theta, u = a_hat.flatten().tolist()

    # 计算历史拟合值
    x1_fit = np.zeros(n)
    x0_fit = np.zeros(n)

    # 第一个数据点不变
    x1_fit[0] = x1[0]
    x0_fit[0] = x0[0]

    # 使用公式(2-8)和(2-9)计算拟合值
    if abs(theta) < 1e-10:
        # 处理θ接近0的情况
        for k in range(1, n):
            x1_fit[k] = x0[0] + u * k
            x0_fit[k] = u
    else:
        # 正常计算公式(2-8)
        for k in range(1, n):
            x1_fit[k] = (x0[0] - u / theta) * np.exp(-theta * (k - 1)) + u / theta
            # 使用公式(2-9)计算还原值
            x0_fit[k] = (1 - np.exp(theta)) * (x0[0] - u / theta) * np.exp(-theta * (k - 1))

    # 计算残差
    residuals = x0 - x0_fit

    return {
        "theta": float(theta),  # 参数θ
        "u": float(u),  # 参数u
        "x0_1": float(x0[0]),  # 原始序列第一个值
        "x0_fit": x0_fit.tolist(),  # 拟合值序列
        "residuals": residuals.tolist(),  # 残差序列
        "x0": x0.tolist()  # 原始序列
    }


def gm11_predict_future_failures(
        params: Dict[str, Union[float, List[float]]],
        failure_intervals: Union[List[float], np.ndarray],
        prediction_step: int = 5,
) -> Dict[str, List[float]]:
    """
    使用 GM(1,1) 模型预测未来失效时间间隔。
    使用公式(2-9): x̂⁽⁰⁾(k+1) = (1 - e^{θ})(x⁽⁰⁾(1) - u/θ)e^{-θk}
    """
    x0 = _to_series(failure_intervals)
    n = x0.size
    theta = params["theta"]
    u = params["u"]
    x0_1 = params["x0_1"]

    # 计算历史拟合值（如果参数中已有则直接使用）
    if "x0_fit" in params:
        x0_fit = np.array(params["x0_fit"])
    else:
        x0_fit = np.zeros(n)
        x0_fit[0] = x0[0]
        if abs(theta) < 1e-10:
            for k in range(1, n):
                x0_fit[k] = u
        else:
            for k in range(1, n):
                x0_fit[k] = (1 - np.exp(theta)) * (x0_1 - u / theta) * np.exp(-theta * (k - 1))

    # 预测未来间隔（使用公式2-9）
    future_intervals: List[float] = []

    if abs(theta) < 1e-10:
        # θ接近0的特殊情况
        for k in range(n, n + prediction_step):
            future_intervals.append(u)
    else:
        # 正常计算公式(2-9)
        for k in range(n, n + prediction_step):
            interval = (1 - np.exp(theta)) * (x0_1 - u / theta) * np.exp(-theta * (k - 1))
            # 确保预测间隔非负
            if interval <= 0:
                interval = 1e-3  # 设为小正数
            future_intervals.append(float(interval))

    # 计算累积失效时间
    cumulative_times = np.cumsum(x0_fit)
    last_time = cumulative_times[-1] if len(cumulative_times) > 0 else 0.0

    # 未来时间点的累积时间
    future_cumulative = []
    current = last_time
    for interval in future_intervals:
        current += interval
        future_cumulative.append(current)

    return {
        "predicted_times": future_cumulative,
        "predicted_intervals": future_intervals,
        "cumulative_times": cumulative_times.tolist(),
        "next_failure_time": future_cumulative[0] if future_cumulative else None,
        "x0_hist_fit": x0_fit.tolist(),
        "full_predicted_intervals": np.concatenate([x0_fit, np.array(future_intervals)]).tolist(),
    }


def calculate_gm11_model_accuracy(
        params: Dict[str, Union[float, List[float]]],
        failure_intervals: Union[List[float], np.ndarray],
) -> Dict[str, Union[float, str]]:
    """
    计算 GM(1,1) 模型精度指标，包括后验方差比值C和小误差概率p。
    根据表2-1判断模型精度等级。

    公式：
        残差 e(k) = X⁽⁰⁾(k) - X̂⁽⁰⁾(k)
        原始序列方差 S1² = (1/N)Σ(X⁽⁰⁾(k) - X̄⁽⁰⁾)²
        残差序列方差 S2² = (1/N)Σ(e(k) - ē)²
        后验方差比值 C = S2 / S1
        小误差概率 p = P(|e(k) - ē| < 0.6745S1)
    """
    # 获取数据
    x0 = _to_series(failure_intervals)

    # 获取拟合值（如果参数中已有）
    if "x0_fit" in params:
        x0_fit = np.array(params["x0_fit"])
    else:
        # 如果没有拟合值，重新计算
        theta = params["theta"]
        u = params["u"]
        x0_1 = params["x0_1"]
        n = x0.size

        x0_fit = np.zeros(n)
        x0_fit[0] = x0[0]

        if abs(theta) < 1e-10:
            for k in range(1, n):
                x0_fit[k] = u
        else:
            for k in range(1, n):
                x0_fit[k] = (1 - np.exp(theta)) * (x0_1 - u / theta) * np.exp(-theta * (k - 1))

    # 计算残差序列（公式2-11）
    residuals = x0 - x0_fit

    # 计算均值（公式2-14）
    x0_mean = np.mean(x0)
    e_mean = np.mean(residuals)

    # 计算方差（公式2-12, 2-13）
    n = x0.size
    S1_squared = np.sum((x0 - x0_mean) ** 2) / n
    S2_squared = np.sum((residuals - e_mean) ** 2) / n

    # 防止除零
    if S1_squared < 1e-10:
        C = float('inf')
    else:
        S1 = np.sqrt(S1_squared)
        S2 = np.sqrt(S2_squared)
        C = S2 / S1  # 公式2-15

    # 计算小误差概率p（公式2-16）
    threshold = 0.6745 * np.sqrt(S1_squared)
    count = np.sum(np.abs(residuals - e_mean) < threshold)
    p = count / n

    # 根据表2-1判断模型精度等级
    if p >= 0.95 and C <= 0.35:
        accuracy_level = "一级（好）"
    elif p >= 0.80 and C <= 0.50:
        accuracy_level = "二级（合格）"
    elif p >= 0.70 and C <= 0.65:
        accuracy_level = "三级（勉强）"
    else:
        accuracy_level = "四级（不合格）"

    return {
        "posterior_variance_ratio_C": float(C),
        "small_error_probability_p": float(p),
        "accuracy_level": accuracy_level,
        "residual_mean": float(e_mean),
        "residual_std": float(np.std(residuals)),
        "S1_squared": float(S1_squared),
        "S2_squared": float(S2_squared),
    }


def calculate_residual_stats(residuals: List[float]) -> Dict[str, float]:
    """计算残差序列的统计特征"""
    if not residuals:
        return {}

    res_array = np.array(residuals)
    return {
        "residual_mean": float(np.mean(res_array)),
        "residual_std": float(np.std(res_array)),
        "residual_max": float(np.max(res_array)),
        "residual_min": float(np.min(res_array)),
        "residual_range": float(np.max(res_array) - np.min(res_array)),
        "residual_median": float(np.median(res_array))
    }


def calculate_gm11_accuracy(
        params: Dict[str, Union[float, List[float]]],
        failure_intervals: Union[List[float], np.ndarray],
) -> Dict[str, float]:
    """
    计算 GM(1,1) 对原始失效时间间隔的预测误差（传统回归指标）。
    保留此函数以兼容原有代码。
    """
    x0 = _to_series(failure_intervals)
    n = x0.size
    if n < 4:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "r2_score": 0.0, "accuracy": 0.0}

    # 获取拟合值
    if "x0_fit" in params:
        x0_fit = np.array(params["x0_fit"])
    else:
        # 重新计算拟合值
        from copy import deepcopy
        train_params = deepcopy(params)
        train_params["x0_fit"] = None
        result = gm11_train_model(failure_intervals)
        x0_fit = np.array(result["x0_fit"])

    # 计算各种精度指标
    mae = float(np.mean(np.abs(x0 - x0_fit)))
    mse = float(np.mean((x0 - x0_fit) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((x0 - x0_fit) ** 2))
    ss_tot = float(np.sum((x0 - np.mean(x0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # 计算相对误差，限制单点最大误差为100%，避免准确率为0%
    rel_err = np.abs((x0 - x0_fit) / (np.abs(x0) + 1e-10))
    rel_err = np.minimum(rel_err, 1.0)  # 限制单点最大相对误差为100%
    acc = float(max(0.0, min(100.0, (1.0 - np.mean(rel_err)) * 100.0)))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2,
        "accuracy": acc,
        "mean_absolute_percentage_error": float(np.mean(rel_err) * 100)
    }


# 使用示例
if __name__ == "__main__":
    # 示例数据
    example_data = [10.2, 12.5, 11.8, 13.2, 14.0, 12.7, 13.5, 14.2]

    try:
        # 训练模型
        print("1. 训练GM(1,1)模型...")
        params = gm11_train_model(example_data)
        print(f"模型参数: θ={params['theta']:.4f}, u={params['u']:.4f}")
        print(f"原始序列第一个值: {params['x0_1']}")

        # 计算模型精度指标
        print("\n2. 计算模型精度指标...")
        accuracy_metrics = calculate_gm11_model_accuracy(params, example_data)
        print(f"后验方差比值 C: {accuracy_metrics['posterior_variance_ratio_C']:.4f}")
        print(f"小误差概率 p: {accuracy_metrics['small_error_probability_p']:.4f}")
        print(f"模型精度等级: {accuracy_metrics['accuracy_level']}")

        # 预测未来失效
        print("\n3. 预测未来失效时间间隔...")
        predictions = gm11_predict_future_failures(params, example_data, prediction_step=3)
        print(f"未来3个失效间隔: {predictions['predicted_intervals']}")
        print(f"对应的时间点: {predictions['predicted_times']}")

        # 计算残差统计
        print("\n4. 残差统计信息...")
        residual_stats = calculate_residual_stats(params['residuals'])
        for key, value in residual_stats.items():
            print(f"{key}: {value:.4f}")

        # 传统精度指标（可选）
        print("\n5. 传统回归指标...")
        reg_metrics = calculate_gm11_accuracy(params, example_data)
        for key, value in reg_metrics.items():
            print(f"{key}: {value:.4f}")

    except ValueError as e:
        print(f"错误: {e}")