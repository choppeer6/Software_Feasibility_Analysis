import numpy as np
from typing import List, Dict, Union, Tuple, Optional
import random

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


def _to_series(failure_data: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    将失效时间列表转为 numpy 序列，保持原始时序。
    
    注意：对于时间序列预测，必须保持原始顺序以保留时序信息。
    如果输入是失效间隔时间，将保持原始顺序；如果是累计时间，也应保持原始顺序。
    不进行排序，以确保时间序列的时序特征得以保留。
    """
    arr = np.asarray(failure_data, dtype=float)
    # 保持原始顺序，不进行排序
    # 这样可以保留时间序列的时序信息，对于SVR等时间序列预测模型至关重要
    return arr


def normalize_data(
    data: Union[List[float], np.ndarray],
    t_min: Optional[float] = None,
    t_max: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    """
    数据归一化：将软件失效时间序列数据转换到(0,1)区间
    
    公式(6.54): t' = (0.8 / (tmax - tmin)) * t + (0.9 - 0.8 * (tmax / (tmax - tmin)))
    
    参数:
        data: 原始失效时间序列数据
        t_min: 最小值（如果为None，则从data中计算）
        t_max: 最大值（如果为None，则从data中计算）
    
    返回:
        normalized_data: 归一化后的数据
        t_min: 原始数据的最小值
        t_max: 原始数据的最大值
    """
    data = _to_series(data)
    
    if t_min is None:
        t_min = float(np.min(data))
    if t_max is None:
        t_max = float(np.max(data))
    
    if t_max == t_min:
        # 如果所有值相同，返回0.9（中间值）
        normalized_data = np.full_like(data, 0.9)
        return normalized_data, t_min, t_max
    
    # 归一化公式(6.54)
    normalized_data = (0.8 / (t_max - t_min)) * data + (0.9 - 0.8 * (t_max / (t_max - t_min)))
    
    return normalized_data, t_min, t_max


def denormalize_data(
    normalized_data: Union[List[float], np.ndarray],
    t_min: float,
    t_max: float
) -> np.ndarray:
    """
    数据反归一化（数据回放）：将归一化后的数据转换回原始尺度
    
    公式(6.57): t = (t' - 0.9) / 0.8 × (tmax - tmin) + tmax
    
    参数:
        normalized_data: 归一化后的数据
        t_min: 原始数据的最小值
        t_max: 原始数据的最大值
    
    返回:
        denormalized_data: 反归一化后的数据
    """
    normalized_data = np.asarray(normalized_data, dtype=float)
    
    # 反归一化公式(6.57)
    denormalized_data = ((normalized_data - 0.9) / 0.8) * (t_max - t_min) + t_max
    
    return denormalized_data


def create_dataset(
    data: Union[List[float], np.ndarray],
    look_back: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将时间序列转换为监督学习格式（X, y）
    例如：look_back=8 时，
        X[0] = [t1, t2, t3, t4, t5, t6, t7, t8] → y[0] = t9
    
    根据图6.11，m值通常在5-15之间，常用值为8
    """
    data = _to_series(data)
    if data.size < look_back + 1:
        raise ValueError(f"SVR 训练数据至少需要 {look_back + 1} 个点")

    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X, dtype=float), np.array(y, dtype=float)


def calculate_fitness(
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
    start_idx: int = 9
) -> float:
    """
    计算适应度函数值
    
    公式(6.55): f = (1 / (n - 8)) * Σ (from i=9 to n) |t'_i - t_i| / t_i
    
    其中:
        t'_i: 归一化后的软件失效时间预测值
        t_i: 归一化后的软件失效时间实测值
        n: 原始数据的总长度
    
    参数:
        actual_values: 归一化后的实际值（从第look_back+1个点开始，对应原始数据的第9个点开始）
        predicted_values: 归一化后的预测值（与actual_values长度相同）
        start_idx: 起始索引（默认9，用于文档说明，实际计算中使用所有数据）
    
    返回:
        fitness: 适应度值（越小越好）
    
    注意:
        如果look_back=8，那么actual_values[0]对应原始数据的第9个点（索引8）
        actual_values的长度n_vals = n_total - 8，正好对应公式中的(n - 8)
    """
    # 确保两个数组长度相同
    min_len = min(len(actual_values), len(predicted_values))
    if min_len == 0:
        return float('inf')
    
    actual_values = actual_values[:min_len]
    predicted_values = predicted_values[:min_len]
    
    n_vals = len(actual_values)
    
    # 计算相对误差
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_errors = np.abs(predicted_values - actual_values) / (actual_values + 1e-10)
    
    # 适应度函数公式(6.55)
    # n_vals = n_total - look_back，如果look_back=8，则n_vals = n_total - 8
    # 正好对应公式中的(n - 8)
    fitness = (1.0 / n_vals) * np.sum(rel_errors)
    
    return float(fitness)


class PSOOptimizer:
    """
    粒子群优化(PSO)算法，用于优化SVR模型的参数C和σ
    
    根据图6.12的流程实现PSO优化
    """
    
    def __init__(
        self,
        n_particles: int = 30,
        max_iterations: int = 100,
        w: float = 0.7,
        c1: float = 2.0,
        c2: float = 2.0,
        C_bounds: Tuple[float, float] = (0.1, 1000.0),
        sigma_bounds: Tuple[float, float] = (0.01, 10.0)
    ):
        """
        初始化PSO优化器
        
        参数:
            n_particles: 粒子数量
            max_iterations: 最大迭代次数
            w: 惯性权重
            c1: 个体学习因子（通常为2）
            c2: 社会学习因子（通常为2）
            C_bounds: C参数的取值范围
            sigma_bounds: σ参数的取值范围
        """
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.C_bounds = C_bounds
        self.sigma_bounds = sigma_bounds
        
        # 粒子位置: [C, sigma]
        self.positions = np.zeros((n_particles, 2))
        # 粒子速度
        self.velocities = np.zeros((n_particles, 2))
        # 个体最优位置
        self.pbest_positions = np.zeros((n_particles, 2))
        # 个体最优适应度
        self.pbest_fitness = np.full(n_particles, float('inf'))
        # 全局最优位置
        self.gbest_position = np.zeros(2)
        # 全局最优适应度
        self.gbest_fitness = float('inf')
        
        # 初始化粒子位置和速度
        self._initialize_particles()
    
    def _initialize_particles(self):
        """初始化粒子位置和速度"""
        for i in range(self.n_particles):
            # 随机初始化位置
            self.positions[i, 0] = random.uniform(self.C_bounds[0], self.C_bounds[1])
            self.positions[i, 1] = random.uniform(self.sigma_bounds[0], self.sigma_bounds[1])
            
            # 初始化个体最优位置
            self.pbest_positions[i] = self.positions[i].copy()
            
            # 随机初始化速度
            self.velocities[i, 0] = random.uniform(-1, 1) * (self.C_bounds[1] - self.C_bounds[0]) * 0.1
            self.velocities[i, 1] = random.uniform(-1, 1) * (self.sigma_bounds[1] - self.sigma_bounds[0]) * 0.1
    
    def optimize(
        self,
        normalized_data: np.ndarray,
        look_back: int = 8,
        epsilon: float = 0.1
    ) -> Tuple[float, float, float]:
        """
        使用PSO优化SVR参数
        
        参数:
            normalized_data: 归一化后的训练数据
            look_back: 回看窗口大小（m值）
            epsilon: SVR的epsilon参数
        
        返回:
            best_C: 最优的C参数
            best_sigma: 最优的σ参数
            best_fitness: 最优适应度值
        """
        X, y = create_dataset(normalized_data, look_back)
        
        for iteration in range(self.max_iterations):
            # 评估每个粒子的适应度
            for i in range(self.n_particles):
                C = self.positions[i, 0]
                sigma = self.positions[i, 1]
                
                # 确保参数在有效范围内
                C = np.clip(C, self.C_bounds[0], self.C_bounds[1])
                sigma = np.clip(sigma, self.sigma_bounds[0], self.sigma_bounds[1])
                
                # 计算gamma（RBF核函数参数）
                # gamma = 1 / (2 * sigma^2)
                gamma = 1.0 / (2.0 * sigma ** 2)
                
                # 训练SVR模型
                try:
                    model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
                    model.fit(X, y)
                    
                    # 预测
                    y_pred = model.predict(X)
                    
                    # 计算适应度
                    fitness = calculate_fitness(y, y_pred, start_idx=9)
                except:
                    fitness = float('inf')
                
                # 更新个体最优位置
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()
                
                # 更新全局最优位置
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.positions[i].copy()
            
            # 更新粒子速度和位置
            for i in range(self.n_particles):
                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)
                
                # 公式(6.56): 更新速度
                self.velocities[i] = (
                    self.w * self.velocities[i] +
                    self.c1 * r1 * (self.pbest_positions[i] - self.positions[i]) +
                    self.c2 * r2 * (self.gbest_position - self.positions[i])
                )
                
                # 公式(6.56): 更新位置
                self.positions[i] = self.positions[i] + self.velocities[i]
                
                # 边界处理
                self.positions[i, 0] = np.clip(self.positions[i, 0], self.C_bounds[0], self.C_bounds[1])
                self.positions[i, 1] = np.clip(self.positions[i, 1], self.sigma_bounds[0], self.sigma_bounds[1])
        
        best_C = float(self.gbest_position[0])
        best_sigma = float(self.gbest_position[1])
        best_fitness = float(self.gbest_fitness)
        
        return best_C, best_sigma, best_fitness


def svr_train_model(
    failure_data: Union[List[float], np.ndarray],
    look_back: int = 8,
    kernel: str = "rbf",
    C: Optional[float] = None,
    gamma: Optional[Union[str, float]] = None,
    epsilon: float = 0.1,
    use_pso: bool = True,
    pso_n_particles: int = 30,
    pso_max_iterations: int = 50,
) -> Tuple[SVR, Dict[str, float], float, float]:
    """
    训练 SVR 模型，用于失效时间序列预测。
    
    根据图6.11和图6.12的流程：
    1. 数据归一化
    2. 使用PSO优化SVR参数（C和σ）
    3. SVR学习
    
    参数:
        failure_data: 原始失效时间序列数据
        look_back: 回看窗口大小（m值，默认8）
        kernel: 核函数类型（默认'rbf'）
        C: 惩罚因子（如果为None且use_pso=True，则使用PSO优化）
        gamma: 核函数参数（如果为None且use_pso=True，则使用PSO优化）
        epsilon: SVR的epsilon参数
        use_pso: 是否使用PSO优化参数
        pso_n_particles: PSO粒子数量
        pso_max_iterations: PSO最大迭代次数
    
    返回:
        model: 训练好的SVR模型（在归一化数据上训练）
        train_metrics: 训练集上的误差指标
        t_min: 原始数据的最小值（用于反归一化）
        t_max: 原始数据的最大值（用于反归一化）
    """
    series = _to_series(failure_data)
    if series.size < look_back + 1:
        raise ValueError(f"SVR 建议至少提供 {look_back + 1} 个数据点")
    
    # 步骤1: 数据归一化（公式6.54）
    normalized_data, t_min, t_max = normalize_data(series)
    
    # 步骤2: 创建数据集
    X, y = create_dataset(normalized_data, look_back)
    
    # 步骤3: 参数优化（使用PSO或手动指定）
    if use_pso and (C is None or gamma is None):
        # 使用PSO优化参数
        pso = PSOOptimizer(
            n_particles=pso_n_particles,
            max_iterations=pso_max_iterations
        )
        best_C, best_sigma, best_fitness = pso.optimize(
            normalized_data,
            look_back=look_back,
            epsilon=epsilon
        )
        # 将sigma转换为gamma
        best_gamma = 1.0 / (2.0 * best_sigma ** 2)
        C = best_C
        gamma = best_gamma
    else:
        # 使用手动指定的参数
        if C is None:
            C = 100.0
        if gamma is None or gamma == 'scale':
            # 如果没有指定gamma，使用默认值
            gamma = 'scale'
    
    # 步骤4: 训练SVR模型（在归一化数据上）
    model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
    model.fit(X, y)
    
    # 步骤5: 计算训练集误差（在归一化数据上）
    y_pred = model.predict(X)
    
    # 计算归一化数据上的误差指标
    mae = float(np.mean(np.abs(y - y_pred)))
    mse = float(np.mean((y - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_score = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    
    # 使用对称相对误差计算准确率
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = np.abs(y) + np.abs(y_pred) + 1e-10
        rel_err = np.abs(y - y_pred) / denominator
        rel_err = np.minimum(rel_err, 1.0)
        mape = float(np.mean(rel_err))
        accuracy = float(max(0.0, min(100.0, (1.0 - mape) * 100.0)))
    
    train_metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2_score,
        "accuracy": accuracy,
    }
    
    return model, train_metrics, t_min, t_max


def svr_predict_future_failures(
    model: SVR,
    failure_data: Union[List[float], np.ndarray],
    prediction_step: int = 5,
    look_back: int = 8,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> Dict[str, List[float]]:
    """
    使用训练好的 SVR 模型预测未来失效时间间隔。
    
    根据图6.11和图6.12的流程：
    1. 数据归一化
    2. SVR预测（在归一化数据上）
    3. 数据回放（反归一化，公式6.57）
    
    注意：模型是在归一化数据上训练的，所以预测也需要在归一化数据上进行
    
    参数:
        model: 训练好的SVR模型（在归一化数据上训练）
        failure_data: 原始失效时间序列数据
        prediction_step: 预测步数
        look_back: 回看窗口大小（m值）
        t_min: 原始数据的最小值（用于归一化，如果为None则从failure_data计算）
        t_max: 原始数据的最大值（用于归一化，如果为None则从failure_data计算）
    
    返回:
        predicted_times: 预测的累计失效时间点
        predicted_intervals: 预测的失效间隔时间
        cumulative_times: 累计失效时间（与predicted_times相同）
        next_failure_time: 下一个失效时间点
    """
    series = _to_series(failure_data)
    if series.size < look_back:
        raise ValueError(f"SVR 预测至少需要 {look_back} 个历史点")
    
    # 计算归一化参数（如果未提供）
    if t_min is None:
        t_min = float(np.min(series))
    if t_max is None:
        t_max = float(np.max(series))
    
    # 步骤1: 数据归一化
    normalized_data, _, _ = normalize_data(series, t_min=t_min, t_max=t_max)
    
    # 计算当前累计失效时间（所有历史间隔时间的总和）
    current_cumulative_time = float(np.sum(series))
    
    # 使用最后 look_back 个归一化后的间隔时间作为初始输入
    current_seq = normalized_data[-look_back:].copy()
    
    predicted_intervals_normalized: List[float] = []
    predicted_intervals: List[float] = []
    cumulative_times: List[float] = []
    current_cumulative = current_cumulative_time
    
    for step in range(prediction_step):
        # 步骤2: SVR预测（在归一化数据上）
        X_input = current_seq.reshape(1, -1)
        next_interval_normalized = float(model.predict(X_input)[0])
        
        # 确保预测值在合理范围内
        if next_interval_normalized <= 0:
            # 如果预测值为负或零，使用历史数据的中位数
            median_normalized = float(np.median(normalized_data))
            next_interval_normalized = max(0.01, median_normalized)
        
        predicted_intervals_normalized.append(next_interval_normalized)
        
        # 步骤3: 数据回放（反归一化，公式6.57）
        next_interval = float(denormalize_data(
            np.array([next_interval_normalized]),
            t_min=t_min,
            t_max=t_max
        )[0])
        
        # 确保预测值为正数
        if next_interval <= 0:
            median_interval = float(np.median(series))
            next_interval = max(0.1, median_interval)
        
        # 如果预测值异常大，进行限制
        max_reasonable = float(np.max(series)) * 2
        if next_interval > max_reasonable:
            next_interval = float(np.max(series)) * 1.5
        
        if next_interval < 0.01:
            next_interval = 0.01
        
        predicted_intervals.append(float(next_interval))
        
        # 累计失效时间 = 当前累计时间 + 预测的间隔
        current_cumulative += next_interval
        cumulative_times.append(float(current_cumulative))
        
        # 更新序列（滑动窗口，使用归一化数据）
        current_seq = np.append(current_seq[1:], next_interval_normalized)
    
    # predicted_times 保持为累计时间点（向后兼容）
    predicted_times = cumulative_times.copy()
    
    return {
        "predicted_times": predicted_times,
        "predicted_intervals": predicted_intervals,
        "cumulative_times": cumulative_times,
        "next_failure_time": cumulative_times[0] if cumulative_times else None,
    }


def calculate_svr_accuracy(
    model: SVR,
    failure_data: Union[List[float], np.ndarray],
    look_back: int = 8,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> Dict[str, float]:
    """
    采用"滚动窗口一步预测"的方式评估 SVR 在历史数据上的误差。
    
    注意：模型是在归一化数据上训练的，所以评估也需要在归一化数据上进行
    
    参数:
        model: 训练好的SVR模型（在归一化数据上训练）
        failure_data: 原始失效时间序列数据
        look_back: 回看窗口大小（m值）
        t_min: 原始数据的最小值（用于归一化，如果为None则从failure_data计算）
        t_max: 原始数据的最大值（用于归一化，如果为None则从failure_data计算）
    
    返回:
        包含mae, mse, rmse, r2_score, accuracy的字典
    """
    series = _to_series(failure_data)
    n = series.size
    if n < look_back + 2:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "r2_score": 0.0, "accuracy": 0.0}
    
    # 计算归一化参数（如果未提供）
    if t_min is None:
        t_min = float(np.min(series))
    if t_max is None:
        t_max = float(np.max(series))
    
    # 数据归一化
    normalized_data, _, _ = normalize_data(series, t_min=t_min, t_max=t_max)
    
    # 创建数据集（在归一化数据上）
    X, y = create_dataset(normalized_data, look_back)
    
    # 预测（在归一化数据上）
    y_pred = model.predict(X)
    
    # 计算归一化数据上的误差指标
    mae = float(np.mean(np.abs(y - y_pred)))
    mse = float(np.mean((y - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_score = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    
    # 使用对称相对误差计算准确率
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = np.abs(y) + np.abs(y_pred) + 1e-10
        rel_err = np.abs(y - y_pred) / denominator
        rel_err = np.minimum(rel_err, 1.0)
        mape = float(np.mean(rel_err))
        accuracy = float(max(0.0, min(100.0, (1.0 - mape) * 100.0)))
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2_score,
        "accuracy": accuracy,
    }