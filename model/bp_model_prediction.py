# -*- coding: utf-8 -*-
"""
基于BP神经网络的软件可靠性预测模型
功能：使用给定的失效时间序列，训练一个三层BP神经网络，预测下一个失效时间点
改进版本：支持模块化调用，可集成到Flask应用中
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import io
from typing import List, Tuple, Dict, Optional, Union

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BPNeuralNetwork:
    """BP神经网络类"""
    
    def __init__(self, input_size=5, hidden_size=10, output_size=1, lr=0.01, random_seed=42):
        """
        初始化网络参数
        input_size: 输入层神经元数（默认5，对应look_back）
        hidden_size: 隐含层神经元数（默认10）
        output_size: 输出层神经元数（默认1，预测一个时间点）
        lr: 学习率，控制参数更新步长
        random_seed: 随机种子，保证结果可复现
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.random_seed = random_seed

        # 设置随机种子
        np.random.seed(random_seed)

        # 初始化权重和偏置（范围[-1, 1]）
        self.W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.b1 = np.random.uniform(-1, 1, (1, hidden_size))
        self.W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.b2 = np.random.uniform(-1, 1, (1, output_size))
        
        # 归一化参数
        self.min_val = None
        self.max_val = None

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数：将输入压缩到(0,1)区间"""
        x = np.clip(x, -500, 500)  # 防止exp溢出
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid函数的导数"""
        return x * (1 - x)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播：计算网络输出"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2
        return self.a2

    def backward(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """反向传播：计算梯度并更新参数"""
        m = X.shape[0]
        d_z2 = self.a2 - y_true.reshape(-1, 1)
        d_W2 = np.dot(self.a1.T, d_z2) / m
        d_b2 = np.mean(d_z2, axis=0, keepdims=True)
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self.sigmoid_derivative(self.a1)
        d_W1 = np.dot(X.T, d_z1) / m
        d_b1 = np.mean(d_z1, axis=0, keepdims=True)
        
        # 更新参数
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, verbose: bool = False) -> List[float]:
        """
        训练网络
        epochs: 训练轮数
        verbose: 是否打印训练过程
        返回：每轮的损失值列表
        """
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = np.mean((y_pred.flatten() - y) ** 2)
            losses.append(loss)
            self.backward(X, y)
            if verbose and epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测接口：输入X，返回预测值（一维数组）"""
        return self.forward(X).flatten()
    
    def normalize(self, data: Union[List[float], np.ndarray],
                  min_val: Optional[float] = None,
                  max_val: Optional[float] = None) -> np.ndarray:
        """归一化数据"""
        if min_val is None:
            min_val = self.min_val
        if max_val is None:
            max_val = self.max_val
        if min_val is None or max_val is None:
            raise ValueError("归一化参数未设置")
        return (data - min_val) / (max_val - min_val)
    
    def denormalize(self, data_norm: Union[List[float], np.ndarray],
                    min_val: Optional[float] = None,
                    max_val: Optional[float] = None) -> np.ndarray:
        """反归一化数据"""
        if min_val is None:
            min_val = self.min_val
        if max_val is None:
            max_val = self.max_val
        if min_val is None or max_val is None:
            raise ValueError("归一化参数未设置")
        return data_norm * (max_val - min_val) + min_val


def create_dataset(data: Union[List[float], np.ndarray],
                   look_back: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    将时间序列转换为监督学习格式（X, y）
    输入：一个时间序列 data = [t1, t2, t3, ..., tn]
    输出：X（输入特征），y（目标值）
    例如：look_back=5 时，
        X[0] = [t1, t2, t3, t4, t5] → y[0] = t6
        X[1] = [t2, t3, t4, t5, t6] → y[1] = t7
    """
    if not isinstance(data, (list, np.ndarray)):
        raise ValueError("data参数必须是列表或numpy数组")
    
    data = np.array(data)
    
    if len(data) < look_back + 1:
        raise ValueError(f"数据长度至少需要 {look_back + 1} 个点才能创建数据集")
    
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)


def bp_train_model(train_data: Union[List[float], np.ndarray],
                   look_back: int = 5,
                   hidden_size: int = 10,
                   lr: float = 0.05,
                   epochs: int = 1500,
                   verbose: bool = False
                   ) -> Tuple[BPNeuralNetwork, float, float, List[float]]:
    """
    训练BP神经网络模型
    
    参数:
        train_data: 训练数据（失效时间序列）
        look_back: 滑动窗口大小（默认5）
        hidden_size: 隐含层神经元数（默认10）
        lr: 学习率（默认0.05）
        epochs: 训练轮数（默认1500）
        verbose: 是否打印训练过程
    
    返回:
        model: 训练好的模型
        min_val: 归一化最小值
        max_val: 归一化最大值
        train_losses: 训练损失列表
    """
    if not isinstance(train_data, (list, np.ndarray)):
        raise ValueError("train_data必须是列表或numpy数组")
    
    train_data = np.array(train_data, dtype=float)
    
    if len(train_data) < look_back + 1:
        raise ValueError(f"训练数据至少需要 {look_back + 1} 个点")
    
    # 创建数据集
    X_train, y_train = create_dataset(train_data, look_back)
    
    # 归一化
    min_val = np.min(train_data)
    max_val = np.max(train_data)
    
    if min_val == max_val:
        raise ValueError("训练数据的所有值相同，无法进行归一化")
    
    X_train_norm = (X_train - min_val) / (max_val - min_val)
    y_train_norm = (y_train - min_val) / (max_val - min_val)
    
    # 创建并训练模型
    model = BPNeuralNetwork(input_size=look_back, hidden_size=hidden_size, 
                           output_size=1, lr=lr)
    model.min_val = min_val
    model.max_val = max_val
    
    train_losses = model.train(X_train_norm, y_train_norm, epochs=epochs, verbose=verbose)
    
    return model, min_val, max_val, train_losses


def bp_predict_future_failures(model: BPNeuralNetwork,
                               train_data: Union[List[float], np.ndarray],
                               prediction_step: int = 5,
                               look_back: int = 5
                               ) -> Dict[str, List[float]]:
    """
    使用训练好的BP模型预测未来失效时间
    
    参数:
        model: 训练好的BP神经网络模型
        train_data: 历史失效时间序列
        prediction_step: 预测步数（默认5）
        look_back: 滑动窗口大小（默认5，需与训练时一致）
    
    返回:
        dict: 包含预测结果的字典
            - predicted_times: 预测的失效时间点列表
            - predicted_intervals: 预测的失效时间间隔列表
            - cumulative_times: 累积时间列表
            - next_failure_time: 下一个失效时间
    """
    if not isinstance(train_data, (list, np.ndarray)):
        raise ValueError("train_data必须是列表或numpy数组")
    
    train_data = np.array(train_data, dtype=float)
    
    if len(train_data) < look_back:
        raise ValueError(f"历史数据至少需要 {look_back} 个点才能进行预测")
    
    predicted_times: List[float] = []
    predicted_intervals: List[float] = []
    cumulative_times: List[float] = []
    
    # 使用最后look_back个点作为初始输入
    current_sequence = train_data[-look_back:].copy()
    current_time = train_data[-1] if len(train_data) > 0 else 0
    
    # 校验模型归一化参数
    if model.min_val is None or model.max_val is None:
        raise ValueError("模型归一化参数 min_val/max_val 未设置，请先使用 bp_train_model 进行训练")
    if model.max_val == model.min_val:
        raise ValueError("模型归一化范围为0，无法进行预测")
    
    for i in range(prediction_step):
        # 归一化当前序列
        X_input = (current_sequence - model.min_val) / (model.max_val - model.min_val)
        X_input = X_input.reshape(1, -1)
        
        # 预测下一个时间点（归一化值）
        y_pred_norm = float(model.predict(X_input)[0])
        
        # 反归一化
        next_time = y_pred_norm * (model.max_val - model.min_val) + model.min_val
        
        # 确保预测值合理（不能小于当前时间）
        if next_time <= current_time:
            next_time = current_time + 1.0  # 至少增加1
        
        predicted_times.append(float(next_time))
        interval = float(next_time - current_time)
        predicted_intervals.append(interval)
        cumulative_times.append(float(next_time))
        
        # 更新序列（滑动窗口）
        current_sequence = np.append(current_sequence[1:], next_time)
        current_time = next_time
    
    return {
        'predicted_times': predicted_times,
        'predicted_intervals': predicted_intervals,
        'cumulative_times': cumulative_times,
        'next_failure_time': predicted_times[0] if len(predicted_times) > 0 else None
    }


def calculate_model_accuracy(model: BPNeuralNetwork,
                             train_data: Union[List[float], np.ndarray],
                             test_data: Optional[Union[List[float], np.ndarray]] = None,
                             look_back: int = 5
                             ) -> Dict[str, float]:
    """
    计算模型预测准确率
    
    参数:
        model: 训练好的BP神经网络模型
        train_data: 训练数据
        test_data: 测试数据（可选，如果提供则计算测试集准确率）
        look_back: 滑动窗口大小
    
    返回:
        dict: 包含准确率指标的字典
            - mae: 平均绝对误差
            - mse: 均方误差
            - rmse: 均方根误差
            - r2_score: R²决定系数
            - accuracy: 准确率（基于相对误差）
    """
    if not isinstance(train_data, (list, np.ndarray)):
        raise ValueError("train_data必须是列表或numpy数组")
    
    train_data = np.array(train_data, dtype=float)
    
    # 如果没有测试数据，使用训练数据的后一部分作为验证
    if test_data is None:
        if len(train_data) < look_back * 2 + 1:
            # 数据太少，无法划分验证集
            return {
                'mae': 0.0,
                'mse': 0.0,
                'rmse': 0.0,
                'r2_score': 0.0,
                'accuracy': 0.0
            }
        # 使用后30%作为验证集
        split_idx = int(len(train_data) * 0.7)
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
    else:
        val_data = np.array(test_data, dtype=float)
    
    if len(val_data) < look_back + 1:
        return {
            'mae': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'r2_score': 0.0,
            'accuracy': 0.0
        }
    
    # 创建验证集
    X_val, y_val = create_dataset(val_data, look_back)
    
    # 归一化
    if model.min_val is None or model.max_val is None or model.max_val == model.min_val:
        # 归一化参数异常时，直接在原始尺度上评价
        X_val_norm = X_val
        y_val_real = y_val
        y_pred_real = model.predict(X_val_norm)
    else:
        X_val_norm = (X_val - model.min_val) / (model.max_val - model.min_val)
        # 预测（在归一化空间）
        y_pred_norm = model.predict(X_val_norm)
        # 反归一化
        y_val_real = y_val
        y_pred_real = y_pred_norm * (model.max_val - model.min_val) + model.min_val
    
    # 计算指标
    mae = np.mean(np.abs(y_val_real - y_pred_real))
    mse = np.mean((y_val_real - y_pred_real) ** 2)
    rmse = np.sqrt(mse)
    
    # R²决定系数
    ss_res = np.sum((y_val_real - y_pred_real) ** 2)
    ss_tot = np.sum((y_val_real - np.mean(y_val_real)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # 准确率（基于相对误差）
    relative_errors = np.abs((y_val_real - y_pred_real) / (y_val_real + 1e-10))
    accuracy = (1 - np.mean(relative_errors)) * 100
    accuracy = max(0, min(100, accuracy))  # 限制在0-100之间
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2_score': float(r2_score),
        'accuracy': float(accuracy)
    }


def plot_prediction_results(train_data, prediction_results, save_path=None):
    """
    绘制预测结果并保存为图片
    
    参数:
        train_data: 训练数据
        prediction_results: 预测结果字典
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(14, 6))
    
    # 子图1：训练损失曲线（如果有）
    if 'train_losses' in prediction_results:
        plt.subplot(1, 2, 1)
        plt.plot(prediction_results['train_losses'], color='b')
        plt.title('Training Loss (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
    
    # 子图2：预测值 vs 真实值（如果有测试数据）
    if 'test_data' in prediction_results and 'test_predictions' in prediction_results:
        plt.subplot(1, 2, 2)
        test_data = prediction_results['test_data']
        test_pred = prediction_results['test_predictions']
        plt.plot(test_data, 'o-', label='True Value', color='green', markersize=8)
        plt.plot(test_pred, 'x--', label='Predicted Value', color='red', markersize=8)
        plt.title('Prediction vs True (Test Set)')
        plt.xlabel('Sample Index')
        plt.ylabel('Failure Time')
        plt.legend()
        plt.grid(True)
    else:
        # 如果没有测试数据，显示历史数据和预测数据
        plt.subplot(1, 2, 2)
        train_array = np.array(train_data)
        predicted_times = prediction_results.get('predicted_times', [])
        
        if len(train_array) > 0:
            plt.plot(range(1, len(train_array) + 1), train_array, 
                    'o-', label='Historical Data', color='blue', markersize=6)
        
        if len(predicted_times) > 0:
            start_idx = len(train_array) + 1
            plt.plot(range(start_idx, start_idx + len(predicted_times)), predicted_times,
                    'x--', label='Predicted Data', color='red', markersize=8)
        
        plt.title('Historical vs Predicted Failure Times')
        plt.xlabel('Failure Sequence')
        plt.ylabel('Failure Time')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存为 {save_path}")
    else:
        plt.savefig('bp_prediction_result.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 'bp_prediction_result.png'")
    
    plt.close()
    
def run_bp_prediction_pipeline(
    train_data: Union[List[float], np.ndarray],
    prediction_step: int = 5,
    look_back: int = 5,
    hidden_size: int = 10,
    lr: float = 0.05,
    epochs: int = 1500,
    test_data: Optional[Union[List[float], np.ndarray]] = None,
    verbose: bool = False
) -> Dict[str, object]:
    """
    便捷封装：完成 BP 模型的「训练 + 未来预测 + 准确率评估」完整流程。

    参数:
        train_data: 训练用失效时间序列
        prediction_step: 需要向前预测的失效点个数
        look_back: 滑动窗口大小
        hidden_size: 隐含层神经元数
        lr: 学习率
        epochs: 训练轮数
        test_data: 可选测试集（若不提供则内部自动切分训练数据）
        verbose: 是否在训练过程中输出损失

    返回:
        dict:
            - model: 训练好的 BPNeuralNetwork 实例
            - min_val / max_val: 归一化参数
            - train_losses: 训练损失列表
            - prediction_results: 未来预测结果（同 bp_predict_future_failures）
            - accuracy_metrics: 准确率指标（同 calculate_model_accuracy）
    """
    model, min_val, max_val, train_losses = bp_train_model(
        train_data,
        look_back=look_back,
        hidden_size=hidden_size,
        lr=lr,
        epochs=epochs,
        verbose=verbose
    )
    
    prediction_results = bp_predict_future_failures(
        model,
        train_data,
        prediction_step=prediction_step,
        look_back=look_back
    )
    
    accuracy_metrics = calculate_model_accuracy(
        model,
        train_data,
        test_data=test_data,
        look_back=look_back
    )
    
    return {
        'model': model,
        'min_val': min_val,
        'max_val': max_val,
        'train_losses': train_losses,
        'prediction_results': prediction_results,
        'accuracy_metrics': accuracy_metrics
    }
