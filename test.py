import numpy as np

def preprocess_data(data, train_stats=None, is_train=True):
    """
    数据预处理：处理缺失值和异常值，训练数据计算统计量，预测数据使用训练统计量
    data: 输入数据（二维列表，每行对应一条数据，列：[写入次数, 读取次数, 平均写入延迟, 平均读取延迟, 设备使用年限]）
    train_stats: 训练数据的统计量（均值、中位数、标准差），is_train=False时需传入
    is_train: 是否为训练数据（True/False）
    return: 预处理后的数据，训练数据时额外返回统计量
    """
    data = np.array(data, dtype=np.float64)
    n_features = data.shape[1]
    stats = {}  # 存储训练数据的统计量：mean（均值）、median（中位数）、std（标准差）
    
    if is_train:
        # 计算训练数据各字段的均值、中位数、标准差（忽略NaN）
        for i in range(n_features):
            valid = data[~np.isnan(data[:, i]), i]
            stats[f'mean_{i}'] = np.mean(valid)
            stats[f'median_{i}'] = np.median(valid)
            stats[f'std_{i}'] = np.std(valid) if len(valid) > 1 else 1.0  # 避免标准差为0
    else:
        # 预测数据使用训练数据的统计量
        stats = train_stats
    
    # 处理缺失值（用均值填充）
    for i in range(n_features):
        data[np.isnan(data[:, i]), i] = stats[f'mean_{i}']
    
    # 处理异常值（用中位数填充）
    # 特征0：写入次数，特征1：读取次数（异常值<0）
    for i in [0, 1]:
        data[data[:, i] < 0, i] = stats[f'median_{i}']
    # 特征2：平均写入延迟，特征3：平均读取延迟（异常值<0或>1000）
    for i in [2, 3]:
        mask = (data[:, i] < 0) | (data[:, i] > 1000)
        data[mask, i] = stats[f'median_{i}']
    # 特征4：设备使用年限（异常值<0或>20）
    mask = (data[:, 4] < 0) | (data[:, 4] > 20)
    data[mask, 4] = stats[f'median_4']
    
    # 训练数据标准化（预测数据后续用训练统计量标准化）
    if is_train:
        normalized_data = (data - np.array([stats[f'mean_{i}'] for i in range(n_features)])) / \
                          np.array([stats[f'std_{i}'] for i in range(n_features)])
        return normalized_data, stats
    else:
        return data

def sigmoid(z):
    """sigmoid激活函数，避免数值溢出"""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def train_logistic_regression(X, y, epochs=100, alpha=0.01):
    """
    批量梯度下降训练逻辑回归模型
    X: 标准化后的训练特征（n_samples × n_features）
    y: 训练标签（n_samples × 1）
    epochs: 迭代次数
    alpha: 学习率
    return: 训练好的权重w
    """
    n_samples, n_features = X.shape
    # 初始化权重（含偏置项，故特征维度+1，先给X添加偏置列）
    X_with_bias = np.hstack([np.ones((n_samples, 1)), X])  # (n_samples, n_features+1)
    w = np.zeros((n_features + 1, 1))  # 初始权重全0
    
    for _ in range(epochs):
        # 计算预测概率
        y_pred_prob = sigmoid(np.dot(X_with_bias, w))
        # 计算梯度（批量梯度，使用全部样本）
        gradient = (1 / n_samples) * np.dot(X_with_bias.T, (y_pred_prob - y.reshape(-1, 1)))
        # 更新权重
        w -= alpha * gradient
    
    return w

def predict(w, X_test, train_stats):
    """
    模型预测
    w: 训练好的权重
    X_test: 预处理后的预测特征（未标准化）
    train_stats: 训练数据的统计量（用于标准化）
    return: 预测结果（0/1）
    """
    n_features = X_test.shape[1]
    # 用训练数据的均值和标准差标准化预测特征
    X_test_norm = (X_test - np.array([train_stats[f'mean_{i}'] for i in range(n_features)])) / \
                  np.array([train_stats[f'std_{i}'] for i in range(n_features)])
    # 添加偏置列
    X_test_with_bias = np.hstack([np.ones((X_test_norm.shape[0], 1)), X_test_norm])
    # 计算预测概率并转为标签（≥0.5为1，否则为0）
    y_pred_prob = sigmoid(np.dot(X_test_with_bias, w))
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()
    return y_pred

def main():
    # 读取输入（注意：实际考试中需从标准输入读取，此处模拟输入格式）
    import sys
    input_lines = [line.strip() for line in sys.stdin if line.strip()]
    ptr = 0
    
    # 读取训练数据
    N = int(input_lines[ptr])
    ptr += 1
    train_data = []
    train_labels = []
    for _ in range(N):
        parts = input_lines[ptr].split()
        ptr += 1
        # 提取特征（索引1-5：写入次数、读取次数、平均写入延迟、平均读取延迟、设备使用年限）
        features = [float(p) if p != 'NaN' else np.nan for p in parts[1:6]]
        # 提取标签（索引6：设备状态）
        label = int(parts[6])
        train_data.append(features)
        train_labels.append(label)
    
    # 读取预测数据
    M = int(input_lines[ptr])
    ptr += 1
    test_data = []
    for _ in range(M):
        parts = input_lines[ptr].split()
        ptr += 1
        # 提取特征（同训练数据，状态字段无意义）
        features = [float(p) if p != 'NaN' else np.nan for p in parts[1:6]]
        test_data.append(features)
    
    # 1. 预处理训练数据
    X_train_norm, train_stats = preprocess_data(train_data, is_train=True)
    y_train = np.array(train_labels)
    
    # 2. 训练逻辑回归模型
    w = train_logistic_regression(X_train_norm, y_train, epochs=100, alpha=0.01)
    
    # 3. 预处理预测数据并预测
    X_test_processed = preprocess_data(test_data, train_stats=train_stats, is_train=False)
    y_pred = predict(w, X_test_processed, train_stats)
    
    # 4. 输出预测结果
    for pred in y_pred:
        print(pred)

if __name__ == "__main__":
    main()
# 代码结束



def main():
    import sys
    # 读取输入（第一行：n, m, p, k；第二行：n个专家概率）
    input_lines = [line.strip() for line in sys.stdin if line.strip()]
    if len(input_lines) < 2:
        print("error")
        return
    
    # 解析第一行参数（专家数n、NPU数m、目标NPU数p、目标专家数k）
    try:
        n, m, p, k = map(int, input_lines[0].split())
        # 校验参数范围（题目规定区间[1,10000]）
        if not (1 <= n <= 10000 and 1 <= m <= 10000 and 1 <= p <= 10000 and 1 <= k <= 10000):
            print("error")
            return
    except ValueError:
        print("error")
        return
    
    # 解析第二行专家概率（n个浮点数，区间(0,1)）
    try:
        probs = list(map(float, input_lines[1].split()))
        if len(probs) != n:
            print("error")
            return
        # 校验概率范围（题目规定(0,1)，此处允许微小精度误差）
        for prob in probs:
            if not (0 < prob < 1):
                print("error")
                return
    except ValueError:
        print("error")
        return
    
    # 第一步：校验专家能否平均分配到NPU（n必须被m整除）
    if n % m != 0:
        print("error")
        return
    group_size = n // m  # 每个NPU对应的专家数量（每组专家数）
    
    # 第二步：构建专家组（按NPU分组，记录每组的专家编号、概率及组最大概率）
    groups = []  # 元素格式：(组最大概率, 组内专家列表)，组内专家格式：(专家编号, 专家概率)
    for group_idx in range(m):
        # 计算当前组专家的编号范围（连续编号）
        start_idx = group_idx * group_size
        end_idx = start_idx + group_size
        group_experts = []
        max_prob_in_group = 0.0
        # 遍历组内专家，收集编号、概率并找组内最大概率
        for expert_idx in range(start_idx, end_idx):
            prob = probs[expert_idx]
            group_experts.append((expert_idx, prob))
            if prob > max_prob_in_group:
                max_prob_in_group = prob
        groups.append((max_prob_in_group, group_experts))
    
    # 第三步：筛选概率最大的p个组（目标NPU对应的组）
    # 按组最大概率降序排序，取前p个组
    groups_sorted = sorted(groups, key=lambda x: x[0], reverse=True)
    target_groups = groups_sorted[:p]
    
    # 第四步：收集目标组内的所有专家，形成待选专家池
    candidate_experts = []
    for _, experts in target_groups:
        candidate_experts.extend(experts)
    
    # 校验待选专家数是否足够k个（不足则输出error）
    if len(candidate_experts) < k:
        print("error")
        return
    
    # 第五步：按专家概率降序排序，选择前k个专家，再按编号升序排列
    # 先按概率降序，概率相同则按编号升序（避免概率一致时排序混乱）
    candidate_experts_sorted = sorted(candidate_experts, key=lambda x: (-x[1], x[0]))
    selected_experts = candidate_experts_sorted[:k]
    # 按专家编号升序排列输出
    selected_ids = sorted([expert[0] for expert in selected_experts])
    
    # 第六步：格式化输出（空格分隔，行尾无空格）
    print(' '.join(map(str, selected_ids)))

if __name__ == "__main__":
    main()
