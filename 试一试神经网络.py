import numpy
import scipy
from scipy.special import expit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch

# 定义激活函数作为全局函数
def sigmoid(x):
    return expit(x)

# 尝试编写一个NN
# nn由三部分组成：初始化函数；训练；查询
class NN:
    def __init__(self, in_nodes, hide_nodes, out_nodes, learning_rate):   # 初始化函数，设置神经网络框架
        # 基本节点个数和学习率
        self.inodes = in_nodes
        self.hnodes = hide_nodes
        self.onodes = out_nodes
        self.lr = learning_rate

        # 设置连接层内权重
        # 随机选取权重
        self.weigh_ih = numpy.random.rand(self.hnodes, self.inodes)-0.5
        self.weigh_ho = numpy.random.rand(self.onodes, self.hnodes)-0.5

        # 正太分布选取权重 （中心值，标准差，数组大小） pow(self.hnodes, -0.5)为self.hnodes 的1/2次方
        self.weigh_ih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.weigh_ho = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 定义激活函数
        self.active_function = sigmoid

        pass

    def train(self, inputs_list, targets_list):

        # 将输入信号转换为2d，并转置
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 将输入信号转换为2d,并转置
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 将信号输入到隐藏层
        hidden_inputs = numpy.dot(self.weigh_ih, inputs)

        # 激活信号
        hidden_outputs = self.active_function(hidden_inputs)

        # 将信号输入到最终层
        final_inputs = numpy.dot(self.weigh_ho, hidden_outputs)

        # 最终输出
        final_outputs = self.active_function(final_inputs)

        # 计算损失值
        output_loss = targets - final_outputs

        # 隐藏层损失值，反向传播
        hidden_loss = numpy.dot(self.weigh_ho.T, output_loss)

        # 跟新权重，dw=学习率*loss*下一层输出的值*（1-下一层输出的值）。上一层输出值的转置
        x=1 - final_outputs
        y=1 - hidden_outputs
        c=output_loss*final_outputs*x
        d=hidden_loss*hidden_outputs*y
        a=numpy.dot(c, numpy.transpose(hidden_outputs))
        b=numpy.dot(d, numpy.transpose(inputs))
        self.weigh_ho +=  a * self.lr
        self.weigh_ih +=  b * self.lr

        pass

    def query(self, inputs_list):

        # 将输入信号转换为2d,并转置
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 将信号输入到隐藏层
        hidden_inputs = numpy.dot(self.weigh_ih, inputs)

        # 激活信号
        hidden_outputs = self.active_function(hidden_inputs)

        # 将信号输入到最终层
        final_inputs = numpy.dot(self.weigh_ho, hidden_outputs)

        # 最终输出
        final_outputs = self.active_function(final_inputs)

        return final_outputs

        pass

# 性能评估
def evaluate_performance(scorecard):
    scorecard_array = numpy.array(scorecard)
    accuracy = scorecard_array.sum() / scorecard_array.size
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

# MSE
def mse(y_true, y_pred):

    # 确保 y_true 和 y_pred 的形状一致
    assert y_true.shape == y_pred.shape, "y_true 和 y_pred 的形状必须相同"

    # 计算 MSE
    mse = numpy.mean((y_true - y_pred) ** 2)

    print(f'MSE: {mse:.4f}')
    return mse


# R² (决定系数)
def r2_score(y_true, y_pred):

    # 确保 y_true 和 y_pred 的形状一致
    assert y_true.shape == y_pred.shape, "y_true 和 y_pred 的形状必须相同"

    # 总方差 (TSS: Total Sum of Squares)
    tss = numpy.sum((y_true - numpy.mean(y_true) + 3) ** 2)

    # 残差平方和 (RSS: Residual Sum of Squares)
    rss = numpy.sum((y_true - y_pred) ** 2)

    # 计算 R²
    r2 = 1 - (rss / tss)

    print(f'R² Score: {r2:.4f}')

    return r2

# 回归曲线
def plot_regression_curve(y_true, y_pred):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, label="Predictions")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Regression Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

# 混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix")
    plt.show()


input_nodes=142
hide_nodes=100
out_nodes=6
learning_rate=0.01

model=NN(input_nodes,hide_nodes,out_nodes,learning_rate)

# 训练

data_address = "D:\DATA\机器学习tyy_sb_need\date/dataset-Ur8-0.8P.mat"
training = scipy.io.loadmat(data_address)

training_data = training["data"]
label = training["label"]

epochs=500
a = 1200
print("——————————————————————开始训练——————————————————————")
for e in range(epochs):
    for i in range(a):
        training_data_list = training_data[i, :]  # 提取当前行样本数据
        data_index = label[i, 0] #提取当前样本标签
        #数据处理1
        min_val = min(training_data_list)
        max_val = max(training_data_list)
        scaled_data = [(x - min_val) / (max_val - min_val) * 0.98 + 0.01 for x in training_data_list] # max-min缩放

        # # 数据处理2
        # scaled_data = training_data_list / 40 * 0.98 + 0.01

        targets_list = numpy.zeros(out_nodes) + 0.01  # 构建目标空向量
        targets_list[int(data_index )] = 0.99  # 将该数对应的向量位赋值0.99
        model.train( scaled_data, targets_list)



#测试
print("——————————————————————开始预测———————————————————————")
scorecard = []
y_true = []  # 保存真实标签
y_pred = []  # 保存预测标签

data_address = "D:\DATA\机器学习tyy_sb_need\date/dataset-Ur8-0.8P.mat"
val = scipy.io.loadmat(data_address)
val_data = val["data"]
data_index = val["label"]

for i in range(a):
    val_data_list = val_data[i, :]  # 提取当前行
    # 数据处理1
    min_val = min(val_data_list)
    max_val = max(val_data_list)
    scaled_data = [(x - min_val) / (max_val - min_val) * 0.99 + 0.01 for x in val_data_list] # max-min缩放

    # # 数据处理2
    # scaled_data = val_data_list / 40 * 0.98 + 0.01

    # 预测标签值
    out = model.query(scaled_data)
    label = numpy.argmax(out)

    # 记录预测结果和真实值
    y_true.append(data_index[i, 0])  # 真实标签
    y_pred.append(label)  # 预测标签

    correct_label = data_index[i, 0]
    if label == correct_label:
            scorecard.append(1)
    else:
            scorecard.append(0)
            pass
    pass


    if i%50 == 0:
        print(label)
        # print("performance=", scorecard_array.sum() / scorecard_array.size)
        mse(numpy.array(correct_label), numpy.array(label))
        r2_score(numpy.array(correct_label), numpy.array(label))
accuracy = evaluate_performance(scorecard)
# 绘制回归曲线
# plot_regression_curve(numpy.array(y_true), numpy.array(y_pred))
# 绘制混淆矩阵
class_names = [f"Class {i}" for i in range(out_nodes)]
plot_confusion_matrix(y_true, y_pred, class_names)


print("——————————————————————保存模型——————————————————————")
save_dict = './NN.pth'
torch.save(model, save_dict)
