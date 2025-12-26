# -*- coding: utf-8 -*-
"""
基于BP神经网络的软件可靠性预测模型（教材版本）
根据《软件系统可靠性分析基础与实践》第6章重构
- 支持两层隐含层（第一层500节点，第二层300节点）
- 输入包括5个历史失效时间和时间t
- 输出预测下一个失效时间
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union


class BPNeuralNetworkV2:
    """两层隐含层BP神经网络类（教材版本）"""

    def __init__(self, input_size=6, hidden1_size=500, hidden2_size=300,
                 output_size=1, lr=0.01, random_seed=42):
        """
        初始化网络参数

        参数:
            input_size: 输入层神经元数（默认6：5个历史时间 + 时间t）
            hidden1_size: 第一隐含层神经元数（默认500）
            hidden2_size: 第二隐含层神经元数（默认300）
            output_size: 输出层神经元数（默认1）
            lr: 学习率
            random_seed: 随机种子
        """
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.lr = lr
        self.random_seed = random_seed

        np.random.seed(random_seed)

        # 初始化权重和偏置
        # 输入层 -> 第一隐含层
        self.W1 = np.random.uniform(-0.5, 0.5, (input_size, hidden1_size))
        self.b1 = np.random.uniform(-0.5, 0.5, (1, hidden1_size))

        # 第一隐含层 -> 第二隐含层
        self.W2 = np.random.uniform(-0.5, 0.5, (hidden1_size, hidden2_size))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, hidden2_size))

        # 第二隐含层 -> 输出层
        self.W3 = np.random.uniform(-0.5, 0.5, (hidden2_size, output_size))
        self.b3 = np.random.uniform(-0.5, 0.5, (1, output_size))

        # 归一化参数
        self.min_val = None
        self.max_val = None
        self.min_time = None
        self.max_time = None

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数（单极性）"""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid函数的导数"""
        return x * (1 - x)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        # 第一隐含层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # 第二隐含层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        # 输出层（线性函数）
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.z3  # 线性输出

        return self.a3

    def backward(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """反向传播"""
        m = X.shape[0]
        y_true = y_true.reshape(-1, 1)

        # 输出层误差
        d_z3 = self.a3 - y_true

        # 第二隐含层 -> 输出层
        d_W3 = np.dot(self.a2.T, d_z3) / m
        d_b3 = np.mean(d_z3, axis=0, keepdims=True)

        # 第一隐含层 -> 第二隐含层
        d_a2 = np.dot(d_z3, self.W3.T)
        d_z2 = d_a2 * self.sigmoid_derivative(self.a2)
        d_W2 = np.dot(self.a1.T, d_z2) / m
        d_b2 = np.mean(d_z2, axis=0, keepdims=True)

        # 输入层 -> 第一隐含层
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self.sigmoid_derivative(self.a1)
        d_W1 = np.dot(X.T, d_z1) / m
        d_b1 = np.mean(d_z1, axis=0, keepdims=True)

        # 更新参数
        self.W3 -= self.lr * d_W3
        self.b3 -= self.lr * d_b3
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000,
              verbose: bool = False, target_error: float = 1e-4) -> List[float]:
        """
        训练网络

        参数:
            X: 输入特征
            y: 目标值
            epochs: 最大训练轮数
            verbose: 是否打印训练过程
            target_error: 目标误差（达到此误差提前停止）

        返回:
            每轮的损失值列表
        """
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = np.mean((y_pred.flatten() - y) ** 2)
            losses.append(loss)

            self.backward(X, y)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

            # 达到目标误差提前停止
            if loss < target_error:
                if verbose:
                    print(f"达到目标误差 {target_error}，训练提前停止于 Epoch {epoch}")
                break

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测接口"""
        return self.forward(X).flatten()


def create_dataset_with_time(data: Union[List[float], np.ndarray],
                             look_back: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建包含时间t的训练数据集（教材格式）

    输入格式：[xt-5, xt-4, xt-3, xt-2, xt-1, t] → xt

    参数:
        data: 累积失效时间序列
        look_back: 滑动窗口大小（默认5）

    返回:
        X: 输入特征矩阵 [样本数, 6]（5个历史时间 + 时间t）
        y: 目标值向量
    """
    data = np.array(data, dtype=float)

    if len(data) < look_back + 1:
        raise ValueError(f"数据长度至少需要 {look_back + 1} 个点")

    X, y = [], []
    for i in range(look_back, len(data)):
        # 输入：5个历史失效时间 + 时间t（使用当前索引作为时间）
        x_input = list(data[i - look_back:i])
        x_input.append(float(i + 1))  # 时间t从1开始
        X.append(x_input)
        y.append(data[i])

    return np.array(X, dtype=float), np.array(y, dtype=float)


def bp_train_model_v2(train_data: Union[List[float], np.ndarray],
                      look_back: int = 5,
                      hidden1_size: int = 500,
                      hidden2_size: int = 300,
                      lr: float = 0.01,
                      epochs: int = 5000,
                      verbose: bool = False,
                      target_error: float = 1e-4
                      ) -> Tuple[BPNeuralNetworkV2, Dict[str, float], List[float]]:
    """
    训练两层隐含层BP神经网络模型（教材版本）

    参数:
        train_data: 训练数据（累积失效时间序列）
        look_back: 滑动窗口大小（默认5）
        hidden1_size: 第一隐含层神经元数（默认500）
        hidden2_size: 第二隐含层神经元数（默认300）
        lr: 学习率（默认0.01）
        epochs: 训练轮数（默认5000）
        verbose: 是否打印训练过程
        target_error: 目标误差

    返回:
        model: 训练好的模型
        metrics: 训练指标
    """
    train_data = np.array(train_data, dtype=float)

    if len(train_data) < look_back + 1:
        raise ValueError(f"训练数据至少需要 {look_back + 1} 个点")

    # 创建数据集（包含时间t）
    X_train, y_train = create_dataset_with_time(train_data, look_back)

    # 归一化：分别对失效时间和时间t进行归一化
    # 失效时间归一化
    min_val = np.min(train_data)
    max_val = np.max(train_data)

    if min_val == max_val:
        raise ValueError("训练数据的所有值相同，无法进行归一化")

    # 时间t归一化（从1到len(train_data)）
    min_time = 1.0
    max_time = float(len(train_data))

    # 归一化输入特征（前5个是失效时间，最后1个是时间t）
    X_train_norm = X_train.copy()
    X_train_norm[:, :-1] = (X_train[:, :-1] - min_val) / (max_val - min_val)
    X_train_norm[:, -1] = (X_train[:, -1] - min_time) / (max_time - min_time)

    # 归一化输出
    y_train_norm = (y_train - min_val) / (max_val - min_val)

    # 创建并训练模型
    model = BPNeuralNetworkV2(
        input_size=look_back + 1,  # 5个历史时间 + 时间t
        hidden1_size=hidden1_size,
        hidden2_size=hidden2_size,
        output_size=1,
        lr=lr
    )
    model.min_val = min_val
    model.max_val = max_val
    model.min_time = min_time
    model.max_time = max_time

    train_losses = model.train(X_train_norm, y_train_norm, epochs=epochs,
                               verbose=verbose, target_error=target_error)

    # 计算训练集误差（反归一化后）
    y_pred_norm = model.predict(X_train_norm)
    y_pred = y_pred_norm * (max_val - min_val) + min_val

    mae = np.mean(np.abs(y_train - y_pred))
    mse = np.mean((y_train - y_pred) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_train - y_pred) ** 2)
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2_score': float(r2_score),
        'final_loss': float(train_losses[-1]) if len(train_losses) > 0 else 0.0
    }

    return model, metrics, train_losses


def bp_predict_future_failures_v2(model: BPNeuralNetworkV2,
                                  train_data: Union[List[float], np.ndarray],
                                  prediction_step: int = 5,
                                  look_back: int = 5,
                                  extended_time_range: float = None
                                  ) -> Dict[str, List[float]]:
    """
    使用训练好的BP模型预测未来失效时间（教材版本）

    参数:
        model: 训练好的BP神经网络模型
        train_data: 历史累积失效时间序列
        prediction_step: 预测步数
        look_back: 滑动窗口大小

    返回:
        预测结果字典
    """
    train_data = np.array(train_data, dtype=float)

    if len(train_data) < look_back:
        raise ValueError(f"历史数据至少需要 {look_back} 个点")

    predicted_times: List[float] = []
    predicted_intervals: List[float] = []
    cumulative_times: List[float] = []

    # 使用最后look_back个点作为初始输入
    current_sequence = train_data[-look_back:].copy()
    current_time = float(train_data[-1])
    # 时间t从训练数据的长度开始（因为训练时最后一个样本的时间t就是len(train_data)）
    current_index = len(train_data)

    for i in range(prediction_step):
        # 构造输入：[xt-5, xt-4, xt-3, xt-2, xt-1, t]
        x_input = current_sequence.copy()
        t_value = float(current_index + 1)  # 时间t从1开始编号
        x_input = np.append(x_input, t_value)
        x_input = x_input.reshape(1, -1)

        # 归一化
        x_input_norm = x_input.copy()
        # 失效时间归一化
        x_input_norm[0, :-1] = (x_input[0, :-1] - model.min_val) / (model.max_val - model.min_val)
        # 时间t归一化：使用扩展的归一化范围（允许超出训练范围）
        # 扩展范围：训练时是[1, len(train_data)]，预测时扩展到[1, len(train_data) + prediction_step]
        if extended_time_range is None:
            extended_max_time = model.max_time + prediction_step * 2  # 扩展2倍预测步数
        else:
            extended_max_time = extended_time_range
        time_norm = (t_value - model.min_time) / (extended_max_time - model.min_time)
        # 允许超出1.0（线性外推），但限制在合理范围内
        time_norm = max(0.0, min(time_norm, 2.0))  # 允许最多2倍的范围
        x_input_norm[0, -1] = time_norm

        # 预测（归一化值）
        y_pred_norm = float(model.predict(x_input_norm)[0])

        # 反归一化
        next_time = y_pred_norm * (model.max_val - model.min_val) + model.min_val

        # 确保预测值合理：必须大于当前时间
        # 如果模型预测值小于等于当前时间，说明模型可能预测的是下一个失效时间相对于当前时间的增量
        # 或者模型输出有问题，此时基于历史趋势进行修正
        if next_time <= current_time:
            # 计算最近几个时间间隔的平均值
            if len(train_data) >= 2:
                recent_intervals = np.diff(train_data[-min(5, len(train_data) - 1):])
                if len(recent_intervals) > 0:
                    avg_interval = np.mean(recent_intervals)
                    # 如果模型输出是增量（即使为负），尝试使用它
                    predicted_interval = next_time - current_time
                    if predicted_interval > 0:
                        # 模型预测了正增量，使用它
                        next_time = current_time + predicted_interval
                    else:
                        # 模型预测了负增量或零，使用历史平均间隔
                        next_time = current_time + avg_interval
                else:
                    next_time = current_time + 1.0
            else:
                next_time = current_time + 1.0
        else:
            # 模型预测值大于当前时间，直接使用
            # 但确保增长不会太快（不超过历史最大间隔的2倍）
            if len(train_data) >= 2:
                recent_intervals = np.diff(train_data[-min(5, len(train_data) - 1):])
                if len(recent_intervals) > 0:
                    max_interval = np.max(recent_intervals)
                    predicted_interval = next_time - current_time
                    # 如果预测间隔过大，限制在合理范围内
                    if predicted_interval > max_interval * 2:
                        next_time = current_time + max_interval * 2

        predicted_times.append(float(next_time))
        interval = float(next_time - current_time)
        predicted_intervals.append(interval)
        cumulative_times.append(float(next_time))

        # 更新序列
        current_sequence = np.append(current_sequence[1:], next_time)
        current_time = next_time
        current_index += 1

    return {
        'predicted_times': predicted_times,
        'predicted_intervals': predicted_intervals,
        'cumulative_times': cumulative_times,
        'next_failure_time': predicted_times[0] if len(predicted_times) > 0 else None
    }


def calculate_model_accuracy_v2(model: BPNeuralNetworkV2,
                                train_data: Union[List[float], np.ndarray],
                                test_data: Optional[Union[List[float], np.ndarray]] = None,
                                look_back: int = 5
                                ) -> Dict[str, float]:
    """
    计算模型预测准确率（教材版本）

    参数:
        model: 训练好的BP神经网络模型
        train_data: 训练数据
        test_data: 测试数据（可选）
        look_back: 滑动窗口大小

    返回:
        准确率指标字典
    """
    train_data = np.array(train_data, dtype=float)
    original_train_len = len(train_data)  # 保存原始训练数据长度

    # 如果没有测试数据，使用训练数据的后一部分作为验证
    if test_data is None:
        if len(train_data) < look_back * 2 + 1:
            # 数据太少，无法划分验证集，返回训练集上的指标
            X_train, y_train = create_dataset_with_time(train_data, look_back)
            X_train_norm = X_train.copy()
            X_train_norm[:, :-1] = (X_train[:, :-1] - model.min_val) / (model.max_val - model.min_val)
            X_train_norm[:, -1] = (X_train[:, -1] - model.min_time) / (model.max_time - model.min_time)
            y_pred_norm = model.predict(X_train_norm)
            y_pred = y_pred_norm.flatten() * (model.max_val - model.min_val) + model.min_val

            mae = np.mean(np.abs(y_train - y_pred))
            mse = np.mean((y_train - y_pred) ** 2)
            rmse = np.sqrt(mse)
            ss_res = np.sum((y_train - y_pred) ** 2)
            ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            relative_errors = np.abs((y_train - y_pred) / (y_train + 1e-10))
            accuracy = (1 - np.mean(relative_errors)) * 100
            accuracy = max(0, min(100, accuracy))

            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2_score': float(r2_score),
                'accuracy': float(accuracy)
            }
        # 使用后5个作为验证集（教材做法）
        # 但需要确保验证集有足够的点来创建至少一个样本
        # 需要至少 look_back + 1 个点才能创建一个样本
        val_size = 5
        if len(train_data) < val_size + look_back:
            # 如果数据不够，使用所有可用数据作为验证集
            val_size = max(1, len(train_data) - look_back)

        split_idx = len(train_data) - val_size
        val_data = train_data[split_idx:]
        # 验证集在原始训练数据中的起始位置
        val_start_idx = split_idx + look_back  # 第一个验证样本的时间t索引
    else:
        val_data = np.array(test_data, dtype=float)
        # 测试数据是独立的，时间t从原始训练数据长度+1开始
        val_start_idx = original_train_len + 1

    # 检查验证集是否有足够的点来创建至少一个样本
    if len(val_data) < look_back + 1:
        # 如果验证集太小，使用训练集的后一部分进行评估
        # 但需要确保有足够的点
        if len(train_data) >= look_back + 1:
            # 使用训练集的最后几个点作为验证集
            val_data = train_data[-(look_back + 1):]
            val_start_idx = len(train_data) - (look_back + 1) + look_back
        else:
            # 数据太少，返回训练集上的指标
            X_train, y_train = create_dataset_with_time(train_data, look_back)
            X_train_norm = X_train.copy()
            X_train_norm[:, :-1] = (X_train[:, :-1] - model.min_val) / (model.max_val - model.min_val)
            X_train_norm[:, -1] = (X_train[:, -1] - model.min_time) / (model.max_time - model.min_time)
            y_pred_norm = model.predict(X_train_norm)
            if y_pred_norm.ndim > 1:
                y_pred_norm = y_pred_norm.flatten()
            y_pred = y_pred_norm * (model.max_val - model.min_val) + model.min_val

            mae = np.mean(np.abs(y_train - y_pred))
            mse = np.mean((y_train - y_pred) ** 2)
            rmse = np.sqrt(mse)
            ss_res = np.sum((y_train - y_pred) ** 2)
            ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            relative_errors = np.abs((y_train - y_pred) / (y_train + 1e-10))
            accuracy = (1 - np.mean(relative_errors)) * 100
            accuracy = max(0, min(100, accuracy))

            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2_score': float(r2_score),
                'accuracy': float(accuracy)
            }

    # 手动创建验证集，确保时间t的索引正确
    X_val, y_val = [], []
    # 从 look_back 开始，到 len(val_data) 结束
    for i in range(look_back, len(val_data)):
        x_input = list(val_data[i - look_back:i])
        # 时间t的索引：从val_start_idx开始，每个样本递增1
        t_idx = val_start_idx + (i - look_back)
        x_input.append(float(t_idx))
        X_val.append(x_input)
        y_val.append(val_data[i])

    # 检查是否创建了验证样本
    if len(X_val) == 0:
        # 如果没有创建任何验证样本，返回训练集上的指标
        X_train, y_train = create_dataset_with_time(train_data, look_back)
        X_train_norm = X_train.copy()
        X_train_norm[:, :-1] = (X_train[:, :-1] - model.min_val) / (model.max_val - model.min_val)
        X_train_norm[:, -1] = (X_train[:, -1] - model.min_time) / (model.max_time - model.min_time)
        y_pred_norm = model.predict(X_train_norm)
        if y_pred_norm.ndim > 1:
            y_pred_norm = y_pred_norm.flatten()
        y_pred = y_pred_norm * (model.max_val - model.min_val) + model.min_val

        mae = np.mean(np.abs(y_train - y_pred))
        mse = np.mean((y_train - y_pred) ** 2)
        rmse = np.sqrt(mse)
        ss_res = np.sum((y_train - y_pred) ** 2)
        ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        relative_errors = np.abs((y_train - y_pred) / (y_train + 1e-10))
        accuracy = (1 - np.mean(relative_errors)) * 100
        accuracy = max(0, min(100, accuracy))

        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2_score),
            'accuracy': float(accuracy)
        }

    X_val = np.array(X_val, dtype=float)
    y_val = np.array(y_val, dtype=float)

    # 归一化
    X_val_norm = X_val.copy()
    X_val_norm[:, :-1] = (X_val[:, :-1] - model.min_val) / (model.max_val - model.min_val)
    # 时间t归一化：使用扩展范围（与预测时一致）
    extended_max_time = model.max_time + 10  # 扩展范围，允许超出训练范围
    time_norm = (X_val[:, -1] - model.min_time) / (extended_max_time - model.min_time)
    time_norm = np.clip(time_norm, 0.0, 2.0)  # 允许超出1.0，最多2倍范围
    X_val_norm[:, -1] = time_norm

    # 预测
    y_pred_norm = model.predict(X_val_norm)
    # 确保y_pred_norm是1D数组
    if y_pred_norm.ndim > 1:
        y_pred_norm = y_pred_norm.flatten()
    y_pred = y_pred_norm * (model.max_val - model.min_val) + model.min_val

    # 确保y_val和y_pred长度一致
    if len(y_val) != len(y_pred):
        min_len = min(len(y_val), len(y_pred))
        y_val = y_val[:min_len]
        y_pred = y_pred[:min_len]

    # 计算指标
    mae = np.mean(np.abs(y_val - y_pred))
    mse = np.mean((y_val - y_pred) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_val - y_pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    relative_errors = np.abs((y_val - y_pred) / (y_val + 1e-10))
    accuracy = (1 - np.mean(relative_errors)) * 100
    accuracy = max(0, min(100, accuracy))

    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2_score': float(r2_score),
        'accuracy': float(accuracy)
    }


if __name__ == "__main__":
    # 教材表6.5的示例数据（前45次失效时间）
    textbook_data = [
        500, 800, 1000, 1100, 1210, 1320, 1390, 1500, 1630, 1700,
        1890, 1960, 2010, 2100, 2150, 2230, 2350, 2470, 2500, 3000,
        3050, 3110, 3170, 3230, 3290, 3320, 3350, 3430, 3480, 3495,
        3540, 3560, 3720, 3750, 3795, 3810, 3830, 3855, 3876, 3896,
        3908, 3920, 3950, 3975, 3982
    ]

    print("=" * 60)
    print("BP神经网络模型测试（教材版本）")
    print("=" * 60)

    # 使用前40个数据训练，后5个数据测试（教材做法）
    train_data = textbook_data[:40]
    test_data = textbook_data[40:]

    print(f"\n训练数据点数: {len(train_data)}")
    print(f"测试数据点数: {len(test_data)}")
    print(f"测试数据: {test_data}")

    # 训练模型
    print("\n开始训练模型...")
    model, train_metrics, train_losses = bp_train_model_v2(
        train_data,
        look_back=5,
        hidden1_size=500,
        hidden2_size=300,
        lr=0.01,
        epochs=5000,
        verbose=True,
        target_error=1e-4
    )

    print("\n训练完成！")
    print(f"训练集 MAE: {train_metrics['mae']:.2f}")
    print(f"训练集 MSE: {train_metrics['mse']:.2f}")
    print(f"训练集 RMSE: {train_metrics['rmse']:.2f}")
    print(f"训练集 R2: {train_metrics['r2_score']:.4f}")

    # 测试模型
    print("\n" + "=" * 60)
    print("测试集预测结果（对比教材表6.7）")
    print("=" * 60)

    accuracy_metrics = calculate_model_accuracy_v2(
        model, train_data, test_data=test_data, look_back=5
    )

    print(f"\n测试集 MAE: {accuracy_metrics['mae']:.2f}")
    print(f"测试集 MSE: {accuracy_metrics['mse']:.2f}")
    print(f"测试集 RMSE: {accuracy_metrics['rmse']:.2f}")
    print(f"测试集 R2: {accuracy_metrics['r2_score']:.4f}")
    print(f"测试集准确率: {accuracy_metrics['accuracy']:.2f}%")

    # 详细预测结果
    print("\n详细预测结果:")
    print("-" * 60)
    print(f"{'样本号':<8} {'输入(最后5个)':<35} {'预测值':<12} {'实际值':<12} {'误差':<12}")
    print("-" * 60)

    # 构造测试集输入
    for i in range(len(test_data)):
        if i < 5:
            # 需要从训练数据末尾取
            start_idx = len(train_data) - 5 + i
            input_times = list(train_data[start_idx:]) + list(test_data[:i])
        else:
            input_times = test_data[i - 5:i]

        # 构造输入
        x_input = np.array(input_times + [len(train_data) + i + 1]).reshape(1, -1)
        x_input_norm = x_input.copy()
        x_input_norm[0, :-1] = (x_input[0, :-1] - model.min_val) / (model.max_val - model.min_val)
        x_input_norm[0, -1] = (x_input[0, -1] - model.min_time) / (model.max_time - model.min_time)

        y_pred_norm = float(model.predict(x_input_norm)[0])
        y_pred = y_pred_norm * (model.max_val - model.min_val) + model.min_val
        y_true = test_data[i]
        error = abs(y_pred - y_true)

        input_str = ','.join([f"{t:.0f}" for t in input_times])
        print(f"{i + 36:<8} [{input_str:<33}] {y_pred:<12.2f} {y_true:<12.0f} {error:<12.2f}")

    # 预测未来失效
    print("\n" + "=" * 60)
    print("未来失效预测")
    print("=" * 60)

    prediction_results = bp_predict_future_failures_v2(
        model, textbook_data, prediction_step=5, look_back=5
    )

    print(f"\n下一次失效预测时间: {prediction_results['next_failure_time']:.2f}")
    print("\n未来失效时间预测:")
    for i, t in enumerate(prediction_results['predicted_times']):
        print(f"第{i + 1}次: {t:.2f}")


# 为了兼容性，创建包装函数
def bp_train_model(train_data: Union[List[float], np.ndarray],
                   look_back: int = 5,
                   hidden_size: int = 10,
                   lr: float = 0.05,
                   epochs: int = 1500,
                   verbose: bool = False,
                   **kwargs) -> Tuple[BPNeuralNetworkV2, float, float, List[float]]:
    """
    兼容性包装函数：将 hidden_size 转换为 hidden1_size 和 hidden2_size
    
    参数:
        train_data: 训练数据
        look_back: 滑动窗口大小
        hidden_size: 隐含层神经元数（兼容旧接口）
        lr: 学习率
        epochs: 训练轮数
        verbose: 是否打印训练过程
        **kwargs: 其他参数（如 target_error）
    
    返回:
        model: 训练好的模型
        min_val: 归一化最小值
        max_val: 归一化最大值
        train_losses: 训练损失列表
    """
    # 将 hidden_size 转换为 hidden1_size 和 hidden2_size
    # 使用更合理的映射：hidden_size 直接作为第一隐藏层大小，第二隐藏层为第一层的60%
    # 这样可以避免网络过大导致训练过慢
    hidden1_size = max(10, hidden_size * 10)  # 至少10，hidden_size=10时得到100
    hidden2_size = max(6, int(hidden1_size * 0.6))  # 第二层为第一层的60%
    
    # 如果 kwargs 中提供了 hidden1_size 或 hidden2_size，优先使用
    if 'hidden1_size' in kwargs:
        hidden1_size = kwargs.pop('hidden1_size')
    if 'hidden2_size' in kwargs:
        hidden2_size = kwargs.pop('hidden2_size')
    
    # 设置合理的默认 target_error，如果未提供则使用更宽松的值以加快训练
    if 'target_error' not in kwargs:
        kwargs['target_error'] = 1e-3  # 使用更宽松的目标误差，加快训练
    
    # 调用 v2 版本
    model, metrics, train_losses = bp_train_model_v2(
        train_data,
        look_back=look_back,
        hidden1_size=hidden1_size,
        hidden2_size=hidden2_size,
        lr=lr,
        epochs=epochs,
        verbose=verbose,
        **kwargs
    )
    
    # 返回格式与旧版本兼容：需要返回 (model, min_val, max_val, train_losses)
    # 从模型对象中获取归一化参数
    min_val = model.min_val
    max_val = model.max_val
    
    return model, min_val, max_val, train_losses


bp_predict_future_failures = bp_predict_future_failures_v2
calculate_model_accuracy = calculate_model_accuracy_v2


def plot_prediction_results(train_data, prediction_results, filename='prediction.png'):
    """绘制预测结果（占位函数，保持兼容性）"""
    pass