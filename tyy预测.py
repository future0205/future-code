from typing import final
import h5py
import numpy
import scipy.special

from conda_build.skeletons.cran import target_platform_bash_test_by_sel
from numba.np.arrayobj import numpy_transpose
from scipy.spatial.distance import num_obs_y


# nn由三部分组成：初始化函数；训练；查询
class NN:
    def __init__(self, in_nodes, out_nodes, learning_rate):   # 初始化函数，设置神经网络框架
        # 基本节点个数和学习率
        self.inodes = in_nodes
        self.onodes = out_nodes
        self.lr = learning_rate

        # 设置连接层内权重
        # 随机选取权重
        # self.weigh_io = numpy.random.rand(self.onodes, self.inodes)-0.5

        # 正太分布选取权重 （中心值，标准差，数组大小） pow(self.onodes, -0.5)为self.onodes 的1/2次方
        self.weigh_io = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.inodes))


        # # 定义激活函数
        # self.active_function = lambda x: scipy.special.expit(x)

        pass

    def train(self,inputs_list, targets_list):

        # 将输入信号转换为2d，并转置
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 将信号输入到最终层
        final_inputs = numpy.dot(self.weigh_io, inputs)

        # 最终输出
        # final_outputs = self.active_function(final_inputs)
        final_outputs = final_inputs * 0.99

        # 计算损失值
        output_loss = targets - final_outputs


        x = 1 - final_outputs
        z = output_loss * final_outputs * x
        y = numpy.dot(z, numpy.transpose(inputs))

        # 跟新权重，dw=学习率*loss*下一层输出的值*（1-下一层输出的值）。上一层输出值的转置
        self.weigh_io += self.lr * y
        print(output_loss)

        pass

    def query(self, inputs_list):

        # 将输入信号转换为2d,并转置
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 将信号输入到最终层
        final_inputs = numpy.dot(self.weigh_io, inputs)

        # 最终输出
        # final_outputs = self.active_function(final_inputs)
        final_outputs = final_inputs * 0.99

        return final_outputs

        pass
    pass
pass

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


input_nodes=9
out_nodes=6
learning_rate=0.3

n=NN(input_nodes,out_nodes,learning_rate)


# 训练
for data_index in range(6):
    data_address = "F:\DATA\机器学习tyy_sb_need\机器学习tyy_sb_need\date/train_sample_{}.mat".format(data_index)
    training_data = scipy.io.loadmat(data_address)
    training_data = training_data["data{}".format(data_index)]

    epochs=400
    for e in range(epochs):
        for i in range(50):
            training_data_list = training_data[i, :]  # 提取当前行
            #数据处理1
            min_val = min(training_data_list)
            max_val = max(training_data_list)
            scaled_data = [(x - min_val) / (max_val - min_val) * 0.98 + 0.01 for x in training_data_list] # max-min缩放

            # # 数据处理2
            # scaled_data = training_data_list / 40 * 0.98 + 0.01

            targets_list = numpy.zeros(out_nodes) + 0.01  # 构建目标空向量
            targets_list[int(data_index)] = 0.99  # 将该数对应的向量位赋值0.99
            n.train( scaled_data, targets_list)






# # 测试1
# # 同上，调用函数query
# scorecard = []
# data_address = "D:\DATA\机器学习tyy_sb_need\date/val_sample.mat"
# val_data = scipy.io.loadmat(data_address)
# for data_index in range(1,7):
#     val_data = val_data["data{}".format(data_index)]
#     correct_label = int(data_index)
#     for i in range(50):
#         val_data_list = val_data[i, :]  # 提取当前行
#
#
#         out = NN.query(n,val_data_list)
#         label = numpy.argmax(out)
#         if label == correct_label:
#             scorecard.append(1)
#         else:
#             scorecard.append(0)
#             pass
#         pass
#     scorecard_array=numpy.array(scorecard)
#     print("performance=",scorecard_array.sum()/scorecard_array.size)

#测试2
scorecard = []
for data_index in range(5):
    data_address = "F:\DATA\机器学习tyy_sb_need\机器学习tyy_sb_need\date/val_sample_{}.mat".format(data_index)
    val_data = scipy.io.loadmat(data_address)
    val_data = val_data["data{}".format(data_index)]
    correct_label = int(data_index)
    for i in range(50):
        val_data_list = val_data[i, :]  # 提取当前行
        # 数据处理1
        min_val = min(val_data_list)
        max_val = max(val_data_list)
        scaled_data = [(x - min_val) / (max_val - min_val) * 0.98 + 0.01 for x in val_data_list] # max-min缩放

        # # 数据处理2
        # scaled_data = val_data_list / 40 * 0.98 + 0.01

        out = n.query(scaled_data)
        label = numpy.argmax(out)
        print(label)
        if label == correct_label:
                scorecard.append(1)
        else:
                scorecard.append(0)
                pass
        pass
        scorecard_array=numpy.array(scorecard)
        print("performance=",scorecard_array.sum()/scorecard_array.size)

        mse(numpy.array(correct_label), numpy.array(label))
        r2_score(numpy.array(correct_label), numpy.array(label))


