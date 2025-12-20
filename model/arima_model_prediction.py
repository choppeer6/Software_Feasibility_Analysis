import numpy as np
from typing import List, Dict, Union, Tuple

from statsmodels.tsa.arima.model import ARIMA


def _to_series(failure_data: Union[List[float], np.ndarray]) -> np.ndarray:
    """将失效时间列表转为 numpy 序列，并做简单排序去重。"""
    arr = np.asarray(failure_data, dtype=float)
    arr = np.sort(arr)
    return arr


def arima_train_model(
    failure_data: Union[List[float], np.ndarray],
    order: Tuple[int, int, int] = (1, 1, 1),
) -> Tuple[ARIMA, Dict[str, float]]:
    """
    训练 ARIMA 模型（对失效时间序列进行建模）。

    返回:
        model: 拟合好的 ARIMAResults 对象
        metrics: 训练期的一些指标（如 AIC）
    """
    series = _to_series(failure_data)
    if series.size < 5:
        raise ValueError("ARIMA 建议至少提供 5 个数据点")

    model = ARIMA(series, order=order)
    result = model.fit()

    metrics = {
        "aic": float(result.aic) if result.aic is not None else 0.0,
        "bic": float(getattr(result, "bic", 0.0)),
    }
    return result, metrics


def arima_predict_future_failures(
    model: ARIMA,
    failure_data: Union[List[float], np.ndarray],
    prediction_step: int = 5,
) -> Dict[str, List[float]]:
    """
    使用 ARIMA 模型预测未来失效时间点。

    直接在“失效时间序列”尺度上做一步步预测。
    """
    series = _to_series(failure_data)
    n = series.size

    # 使用动态 forecast，基于最后一个观测点向后预测
    forecast = model.forecast(steps=prediction_step)
    forecast = np.asarray(forecast, dtype=float)

    # 确保预测时间单调递增且不早于当前最后失效时间
    predicted_times: List[float] = []
    current_time = float(series[-1])
    for t_pred in forecast:
        t_val = max(float(t_pred), current_time + 1e-6)
        predicted_times.append(t_val)
        current_time = t_val

    predicted_intervals: List[float] = []
    cumulative_times: List[float] = []
    last = float(series[-1])
    for t_val in predicted_times:
        interval = float(t_val - last)
        predicted_intervals.append(interval)
        cumulative_times.append(t_val)
        last = t_val

    return {
        "predicted_times": predicted_times,
        "predicted_intervals": predicted_intervals,
        "cumulative_times": cumulative_times,
        "next_failure_time": predicted_times[0] if predicted_times else None,
    }


def calculate_arima_accuracy(
    model: ARIMA,
    failure_data: Union[List[float], np.ndarray],
) -> Dict[str, float]:
    """
    简单计算 ARIMA 对历史序列的一步预测误差。
    """
    series = _to_series(failure_data)
    n = series.size
    if n < 6:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "r2_score": 0.0, "accuracy": 0.0}

    # 采用“向前滚动一步预测”的方式估计误差
    train = series[:-1]
    true_next = series[1:]

    # 简化：用同一个模型对整个序列进行一步预测
    preds = model.predict(start=1, end=n - 1)
    preds = np.asarray(preds, dtype=float)

    mae = float(np.mean(np.abs(preds - true_next)))
    mse = float(np.mean((preds - true_next) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((true_next - preds) ** 2))
    ss_tot = float(np.sum((true_next - np.mean(true_next)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    rel_err = np.abs((true_next - preds) / (true_next + 1e-10))
    acc = float(max(0.0, min(100.0, (1.0 - np.mean(rel_err)) * 100.0)))

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2_score": r2, "accuracy": acc}


