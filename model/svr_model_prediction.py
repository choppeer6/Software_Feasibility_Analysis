import numpy as np
from typing import List, Dict, Union, Tuple

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


def _to_series(failure_data: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    将失效时间列表转为 numpy 序列，并按时间排序。
    """
    arr = np.asarray(failure_data, dtype=float)
    arr = np.sort(arr)
    return arr


def create_dataset(
    data: Union[List[float], np.ndarray],
    look_back: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将时间序列转换为监督学习格式（X, y）
    例如：look_back=5 时，
        X[0] = [t1, t2, t3, t4, t5] → y[0] = t6
    """
    data = _to_series(data)
    if data.size < look_back + 1:
        raise ValueError(f"SVR 训练数据至少需要 {look_back + 1} 个点")

    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X, dtype=float), np.array(y, dtype=float)


def svr_train_model(
    failure_data: Union[List[float], np.ndarray],
    look_back: int = 5,
    kernel: str = "rbf",
    C: float = 100.0,
    gamma: Union[str, float] = "scale",
    epsilon: float = 0.1,
) -> Tuple[SVR, StandardScaler, Dict[str, float]]:
    """
    训练 SVR 模型，用于失效时间序列预测。

    返回:
        model: 训练好的 SVR 模型
        scaler: 特征标准化器
        train_metrics: 训练集上的简单误差指标
    """
    series = _to_series(failure_data)
    if series.size < look_back + 1:
        raise ValueError(f"SVR 建议至少提供 {look_back + 1} 个数据点")

    X, y = create_dataset(series, look_back)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
    model.fit(X_scaled, y)

    # 训练集误差
    y_pred = model.predict(X_scaled)
    mae = float(np.mean(np.abs(y - y_pred)))
    mse = float(np.mean((y - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_score = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    rel_err = np.abs((y - y_pred) / (y + 1e-10))
    accuracy = float(max(0.0, min(100.0, (1.0 - np.mean(rel_err)) * 100.0)))

    train_metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2_score,
        "accuracy": accuracy,
    }

    return model, scaler, train_metrics


def svr_predict_future_failures(
    model: SVR,
    scaler: StandardScaler,
    failure_data: Union[List[float], np.ndarray],
    prediction_step: int = 5,
    look_back: int = 5,
) -> Dict[str, List[float]]:
    """
    使用训练好的 SVR 模型预测未来失效时间点。
    """
    series = _to_series(failure_data)
    if series.size < look_back:
        raise ValueError(f"SVR 预测至少需要 {look_back} 个历史点")

    # 使用最后 look_back 个点作为初始输入
    current_seq = series[-look_back:].copy()
    current_time = float(series[-1])

    predicted_times: List[float] = []
    predicted_intervals: List[float] = []
    cumulative_times: List[float] = []

    for _ in range(prediction_step):
        X_input = current_seq.reshape(1, -1)
        X_input_scaled = scaler.transform(X_input)
        next_time = float(model.predict(X_input_scaled)[0])

        # 保证时间单调递增
        if next_time <= current_time:
            next_time = current_time + 1.0

        predicted_times.append(next_time)
        interval = next_time - current_time
        predicted_intervals.append(float(interval))
        cumulative_times.append(next_time)

        current_time = next_time
        # 更新序列
        current_seq = np.append(current_seq[1:], next_time)

    return {
        "predicted_times": [float(t) for t in predicted_times],
        "predicted_intervals": [float(i) for i in predicted_intervals],
        "cumulative_times": [float(t) for t in cumulative_times],
        "next_failure_time": float(predicted_times[0]) if predicted_times else None,
    }


def calculate_svr_accuracy(
    model: SVR,
    scaler: StandardScaler,
    failure_data: Union[List[float], np.ndarray],
    look_back: int = 5,
) -> Dict[str, float]:
    """
    采用“滚动窗口一步预测”的方式评估 SVR 在历史数据上的误差。
    """
    series = _to_series(failure_data)
    n = series.size
    if n < look_back + 2:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "r2_score": 0.0, "accuracy": 0.0}

    X, y = create_dataset(series, look_back)
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)

    mae = float(np.mean(np.abs(y - y_pred)))
    mse = float(np.mean((y - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_score = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    rel_err = np.abs((y - y_pred) / (y + 1e-10))
    accuracy = float(max(0.0, min(100.0, (1.0 - np.mean(rel_err)) * 100.0)))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2_score,
        "accuracy": accuracy,
    }


