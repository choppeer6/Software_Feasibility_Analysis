# -*- coding: utf-8 -*-
"""
基于BP神经网络的软件可靠性预测模型（详细注释版）
功能：使用给定的失效时间序列，训练一个三层BP神经网络，预测下一个失效时间点
作者：课程设计专用
日期：2025年11月
"""

# 导入必要的库
import numpy as np          # 用于数值计算（数组、矩阵运算）
import matplotlib.pyplot as plt  # 用于绘图

# 解决方案：设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# SimHei 是黑体，Windows 系统通常自带。如果找不到 SimHei，会回退到 DejaVu Sans

# 这行是为了解决负号（'-'）显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# =============================================================================
# 第1部分：数据准备 —— 加载失效时间序列并构建训练/测试样本
# =============================================================================

# PPT中提供的45个软件失效时间点（单位：小时）
# 这些是系统在测试过程中发生故障的累计时间
failure_times = [
    500, 800, 1000, 1100, 1210, 1320, 1390, 1500, 1630, 1700,
    1890, 1960, 2010, 2100, 2150, 2230, 2350, 2470, 2500, 3000,
    3050, 3110, 3170, 3230, 3290, 3320, 3350, 3430, 3480, 3495,
    3540, 3560, 3720, 3750, 3795, 3810, 3830, 3855, 3876, 3896,
    3908, 3920, 3950, 3975, 3982
]

# 定义滑动窗口大小：用前5个时间点预测第6个
# 这是时间序列预测的常用方法，称为“监督学习转换”
look_back = 5

# 划分数据：
# - 训练数据：前40个失效时间点（用于训练模型）
# - 测试数据：从第36个开始取10个（因为要构建5个测试样本，每个需要6个点）
#   实际上我们只需要最后5个预测值，所以取 failure_times[35:]（共10个点）即可构建5组输入-输出对
train_data = failure_times[:40]   # 索引0~39，共40个
test_data = failure_times[35:]    # 索引35~44，共10个 → 可生成5组测试样本

# 定义函数：将时间序列转换为监督学习格式（X, y）
def create_dataset(data, look_back=5):
    """
    输入：一个时间序列 data = [t1, t2, t3, ..., tn]
    输出：X（输入特征），y（目标值）
    例如：look_back=5 时，
        X[0] = [t1, t2, t3, t4, t5] → y[0] = t6
        X[1] = [t2, t3, t4, t5, t6] → y[1] = t7
        ...
    """
    X, y = [], []  # 初始化空列表
    # 遍历数据，确保不会越界（最后一个样本是 data[-look_back-1] 到 data[-1]）
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])      # 取连续5个作为输入
        y.append(data[i + look_back])        # 下一个作为输出
    return np.array(X), np.array(y)  # 转为NumPy数组，便于计算

# 调用函数构建训练集和测试集
X_train, y_train = create_dataset(train_data, look_back)  # 形状：(35, 5), (35,)
X_test, y_test = create_dataset(test_data, look_back)     # 形状：(5, 5), (5,)

# 打印数据形状，确认是否正确
print("训练集大小:", X_train.shape)  # 应为 (35, 5)
print("测试集大小:", X_test.shape)   # 应为 (5, 5)

# =============================================================================
# 第2部分：数据归一化（标准化）
# 原因：神经网络对输入尺度敏感，归一化可加速收敛、提高稳定性
# 方法：Min-Max归一化 → 将所有值缩放到 [0, 1] 区间
# 公式：x_norm = (x - min) / (max - min)
# 注意：使用整个数据集（45个点）的 min/max，保证训练/测试在同一尺度
# =============================================================================

min_val = np.min(failure_times)  # 全局最小值：500
max_val = np.max(failure_times)  # 全局最大值：3982

# 对训练集和测试集分别归一化
X_train_norm = (X_train - min_val) / (max_val - min_val)
X_test_norm = (X_test - min_val) / (max_val - min_val)
y_train_norm = (y_train - min_val) / (max_val - min_val)
y_test_norm = (y_test - min_val) / (max_val - min_val)


# =============================================================================
# 第3部分：手动实现BP神经网络类
# 网络结构：输入层(5) → 隐含层(10, Sigmoid激活) → 输出层(1, 线性输出)
# 用于回归任务（预测连续值）
# =============================================================================

class BPNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        """
        初始化网络参数
        input_size: 输入层神经元数（= look_back = 5）
        hidden_size: 隐含层神经元数（可调，这里设为10）
        output_size: 输出层神经元数（=1，预测一个时间点）
        lr: 学习率，控制参数更新步长
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr  # 学习率

        # 设置随机种子，保证每次运行结果一致（便于调试）
        np.random.seed(42)

        # 初始化权重和偏置（范围[-1, 1]，符合课设要求）
        # W1: 输入层到隐含层的权重矩阵，形状 (5, 10)
        self.W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        # b1: 隐含层偏置，形状 (1, 10)
        self.b1 = np.random.uniform(-1, 1, (1, hidden_size))
        # W2: 隐含层到输出层的权重矩阵，形状 (10, 1)
        self.W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        # b2: 输出层偏置，形状 (1, 1)
        self.b2 = np.random.uniform(-1, 1, (1, output_size))

    def sigmoid(self, x):
        """Sigmoid激活函数：将输入压缩到(0,1)区间"""
        # 防止exp溢出（x太大时exp(-x)会变成inf）
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Sigmoid函数的导数：用于反向传播计算梯度
        注意：这里x是sigmoid的输出值，不是原始输入
        公式：σ'(x) = σ(x) * (1 - σ(x))
        """
        return x * (1 - x)

    def forward(self, X):
        """
        前向传播：计算网络输出
        X: 输入数据，形状 (batch_size, input_size)
        """
        # 隐含层计算：线性组合 + 激活
        self.z1 = np.dot(X, self.W1) + self.b1  # (batch, 10)
        self.a1 = self.sigmoid(self.z1)          # (batch, 10)

        # 输出层计算：线性组合（回归任务不用激活函数）
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # (batch, 1)
        self.a2 = self.z2  # 直接输出

        return self.a2

    def backward(self, X, y_true):
        """
        反向传播：计算梯度并更新参数
        y_true: 真实标签，一维数组 (batch_size,)
        """
        m = X.shape[0]  # 样本数量

        # 1. 计算输出层误差（MSE损失对输出的导数）
        # y_pred 是 self.a2，形状 (m, 1)
        d_z2 = self.a2 - y_true.reshape(-1, 1)  # (m, 1)

        # 2. 计算输出层梯度
        d_W2 = np.dot(self.a1.T, d_z2) / m      # (10, 1)
        d_b2 = np.mean(d_z2, axis=0, keepdims=True)  # (1, 1)

        # 3. 误差反传到隐含层
        d_a1 = np.dot(d_z2, self.W2.T)          # (m, 10)
        d_z1 = d_a1 * self.sigmoid_derivative(self.a1)  # (m, 10)

        # 4. 计算隐含层梯度
        d_W1 = np.dot(X.T, d_z1) / m            # (5, 10)
        d_b1 = np.mean(d_z1, axis=0, keepdims=True)  # (1, 10)

        # 5. 更新参数（梯度下降）
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1

    def train(self, X, y, epochs=1000):
        """
        训练网络
        epochs: 训练轮数
        返回：每轮的损失值列表，用于绘图
        """
        losses = []
        for epoch in range(epochs):
            # 前向传播得到预测值
            y_pred = self.forward(X)
            # 计算均方误差（MSE）
            loss = np.mean((y_pred.flatten() - y) ** 2)
            losses.append(loss)
            # 反向传播更新参数
            self.backward(X, y)
            # 每200轮打印一次损失，观察训练进度
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        return losses

    def predict(self, X):
        """预测接口：输入X，返回预测值（一维数组）"""
        return self.forward(X).flatten()


# =============================================================================
# 第4部分：模型训练与预测
# =============================================================================

# 创建BP神经网络实例
# 输入5个时间点 → 隐含层10个神经元 → 输出1个预测值
model = BPNeuralNetwork(input_size=5, hidden_size=10, output_size=1, lr=0.05)

# 开始训练（使用归一化后的数据）
print("开始训练模型...")
train_losses = model.train(X_train_norm, y_train_norm, epochs=1500)

# 在测试集上预测（归一化值）
y_pred_norm = model.predict(X_test_norm)

# 将预测结果和真实值反归一化回原始时间尺度
# 公式：x_original = x_norm * (max - min) + min
y_test_real = y_test  # 测试真实值本来就是原始值
y_pred_real = y_pred_norm * (max_val - min_val) + min_val


# =============================================================================
# 第5部分：输出详细预测结果（每组对比）
# =============================================================================

print("\n" + "="*60)
print("测试集每组预测结果（真实值 vs 预测值）")
print("="*60)
print(f"{'序号':<4} {'真实值':<12} {'预测值':<12} {'绝对误差':<10}")
print("-"*60)
for i in range(len(y_test_real)):
    true_val = y_test_real[i]
    pred_val = y_pred_real[i]
    abs_err = abs(true_val - pred_val)
    print(f"{i+1:<4} {true_val:<12.4f} {pred_val:<12.4f} {abs_err:<10.4f}")
print("-"*60)


# =============================================================================
# 第6部分：模型评估（计算MAE和RMSE）
# =============================================================================

def mae(y_true, y_pred):
    """平均绝对误差：衡量预测值与真实值的平均偏差"""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """均方根误差：对大误差更敏感，常用于回归任务"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# 打印评估结果
print("\n" + "="*40)
print("模型评估结果")
print("="*40)
print(f"MAE: {mae(y_test_real, y_pred_real):.4f}")
print(f"RMSE: {rmse(y_test_real, y_pred_real):.4f}")
print("="*40)


# =============================================================================
# 第7部分：可视化结果
# =============================================================================

plt.figure(figsize=(12, 4))  # 创建一个宽图

# 子图1：训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, color='b')
plt.title('Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# 子图2：预测值 vs 真实值
plt.subplot(1, 2, 2)
plt.plot(y_test_real, 'o-', label='True Value', color='green')
plt.plot(y_pred_real, 'x--', label='Predicted Value', color='red')
plt.title('Prediction vs True')
plt.xlabel('Sample Index')
plt.ylabel('Failure Time')
plt.legend()
plt.grid(True)

# 自动调整布局并保存图片
plt.tight_layout()
plt.savefig('bp_prediction_result.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n图表已保存为 'bp_prediction_result.png'")