import numpy as np
from typing import List, Dict, Union


def _to_series(failure_data: Union[List[float], np.ndarray]) -> np.ndarray:
    """将失效时间列表转为 numpy 序列，并按时间排序。"""
    arr = np.asarray(failure_data, dtype=float)
    arr = np.sort(arr)
    return arr


def gm11_train_model(failure_data: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    训练一阶灰色 GM(1,1) 模型。

    模型形式：
        累加生成序列 X1(k) = sum_{i=1..k} X0(i)
        白化方程 dx/dt + a x = b
        时刻 k 的解 x_hat(k) = (X0(1) - b/a) * exp(-a (k-1)) + b/a
    """
    x0 = _to_series(failure_data)
    n = x0.size
    if n < 4:
        raise ValueError("GM(1,1) 建议至少提供 4 个以上数据点")

    # 1-AGO 累加生成
    x1 = np.cumsum(x0)

    # 邻近均值生成序列 z1
    z1 = 0.5 * (x1[:-1] + x1[1:])

    B = np.column_stack((-z1, np.ones_like(z1)))
    Y = x0[1:].reshape(-1, 1)

    # 最小二乘估计 [a, b]^T
    # theta = (B^T B)^(-1) B^T Y
    BTB = B.T @ B
    if np.linalg.det(BTB) == 0:
        raise ValueError("GM(1,1) 参数矩阵奇异，无法拟合")
    theta = np.linalg.inv(BTB) @ B.T @ Y
    a, b = theta.flatten().tolist()

    return {"a": float(a), "b": float(b), "x0_1": float(x0[0])}


def gm11_predict_future_failures(
    params: Dict[str, float],
    failure_data: Union[List[float], np.ndarray],
    prediction_step: int = 5,
) -> Dict[str, List[float]]:
    """
    使用 GM(1,1) 模型预测未来失效时间。
    返回的是在原始尺度上的预测序列及间隔。
    """
    x0 = _to_series(failure_data)
    n = x0.size
    a = params["a"]
    b = params["b"]
    x0_1 = params["x0_1"]

    # 时刻 k 的累计预测 x1_hat(k)
    def x1_hat(k: int) -> float:
        return (x0_1 - b / a) * np.exp(-a * (k - 1)) + b / a

    # 先重构历史预测（用于计算误差）
    x1_hist = np.array([x1_hat(k) for k in range(1, n + 1)])
    x0_hist = np.diff(x1_hist, prepend=0.0)

    # 预测未来点
    future_x0: List[float] = []
    for k in range(n + 1, n + prediction_step + 1):
        x1_k = x1_hat(k)
        x1_k_1 = x1_hat(k - 1)
        future_x0.append(float(x1_k - x1_k_1))

    # 将预测间隔累加到最后一个失效时间得到未来失效时间点
    last_time = float(x0[-1])
    predicted_times: List[float] = []
    predicted_intervals: List[float] = []
    cumulative_times: List[float] = []
    current_time = last_time
    for interval in future_x0:
        if interval <= 0:
            interval = 1.0
        current_time += interval
        predicted_intervals.append(float(interval))
        predicted_times.append(float(current_time))
        cumulative_times.append(float(current_time))

    return {
        "predicted_times": predicted_times,
        "predicted_intervals": predicted_intervals,
        "cumulative_times": cumulative_times,
        "next_failure_time": predicted_times[0] if predicted_times else None,
        "x0_hist_fit": x0_hist.tolist(),
    }


def calculate_gm11_accuracy(
    params: Dict[str, float],
    failure_data: Union[List[float], np.ndarray],
) -> Dict[str, float]:
    """
    计算 GM(1,1) 对原始数据的一步预测误差。
    """
    x0 = _to_series(failure_data)
    n = x0.size
    if n < 4:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "r2_score": 0.0, "accuracy": 0.0}

    a = params["a"]
    b = params["b"]
    x0_1 = params["x0_1"]

    def x1_hat(k: int) -> float:
        return (x0_1 - b / a) * np.exp(-a * (k - 1)) + b / a

    x1_hat_vals = np.array([x1_hat(k) for k in range(1, n + 1)])
    x0_hat = np.diff(x1_hat_vals, prepend=0.0)

    mae = float(np.mean(np.abs(x0 - x0_hat)))
    mse = float(np.mean((x0 - x0_hat) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((x0 - x0_hat) ** 2))
    ss_tot = float(np.sum((x0 - np.mean(x0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    rel_err = np.abs((x0 - x0_hat) / (x0 + 1e-10))
    acc = float(max(0.0, min(100.0, (1.0 - np.mean(rel_err)) * 100.0)))

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2_score": r2, "accuracy": acc}


