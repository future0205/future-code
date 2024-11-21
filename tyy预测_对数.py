import numpy as np
import scipy
from scipy.io import loadmat
import torch

class Log:
    def __init__(self, learning_rate=0.01, num_iterations=1000):   # 初始化
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):     # 对数函数
        return 1 / (1 + np.exp(-z))

    def one_hot_encode(self, y, num_classes):   # 将类别标签转换为独热编码（one-hot encoding）格式
        return np.eye(num_classes)[y.astype(int)]

    def train(self, x, y):
        num_samples, num_features = x.shape
        num_classes = len(y)
        y_encoded = self.one_hot_encode(y, num_classes)

        # 初始权重
        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros((1, num_classes))

        # 迭代训练
        for i in range(self.num_iterations):
            linear_model = np.dot(x, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # 损失值
            loss = -np.mean(y_encoded * np.log(y_pred + 1e-10))

            dw = (1 / num_samples) * np.dot(x.T, (y_pred - y_encoded))
            db = (1 / num_samples) * np.sum(y_pred - y_encoded, axis=0, keepdims=True)

            # 更新权重和偏置
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:  # 每100次迭代打印一次损失
                print(f'Iteration {i}, Loss: {loss:.4f}')

    def val(self, x):   # 测试
        linear_model = np.dot(x, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return np.argmax(y_pred, axis=1)


# MSE
def mse(y_true, y_pred):

    # 确保 y_true 和 y_pred 的形状一致
    assert y_true.shape == y_pred.shape, "y_true 和 y_pred 的形状必须相同"

    # 计算 MSE
    mse = np.mean((y_true - y_pred) ** 2)

    print(f'MSE: {mse:.4f}')
    return mse


# R² (决定系数)
def r2_score(y_true, y_pred):

    # 确保 y_true 和 y_pred 的形状一致
    assert y_true.shape == y_pred.shape, "y_true 和 y_pred 的形状必须相同"

    # 总方差 (TSS: Total Sum of Squares)
    tss = np.sum((y_true - np.mean(y_true) + 3) ** 2)

    # 残差平方和 (RSS: Residual Sum of Squares)
    rss = np.sum((y_true - y_pred) ** 2)

    # 计算 R²
    r2 = 1 - (rss / tss)

    print(f'R² Score: {r2:.4f}')

    return r2



# 创建并训练模型
model = Log(learning_rate=0.1, num_iterations=1000)

print("——————————————————————开始训练——————————————————————")
for data_index in range(6):
    print("class = {}".format(data_index))
    data_address = "F:\DATA\机器学习tyy_sb_need\机器学习tyy_sb_need\date/train_sample_{}.mat".format(data_index)
    X = loadmat(data_address)
    training_data = X["data{}".format(data_index)]
    targets_list = np.zeros(50) + int(data_index)  # 构建目标空向量, 将该数对应的向量位赋值
    model.train(training_data, targets_list)




# 进行预测
print("——————————————————————开始预测———————————————————————")
for data_index in range(5):
    data_address = "F:\DATA\机器学习tyy_sb_need\机器学习tyy_sb_need\date/val_sample_{}.mat".format(data_index)
    val_data = loadmat(data_address)
    val_data = val_data["data{}".format(data_index)]
    true_list = np.zeros(50) + int(data_index)
    y_pred = model.val(val_data)
    # 输出预测结果
    print(f'预测结果: {y_pred[:50]}')
    mse(true_list, y_pred)
    r2_score(true_list, y_pred)

