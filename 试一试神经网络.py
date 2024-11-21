import numpy
import scipy.special
from conda_build.skeletons.cran import target_platform_bash_test_by_sel
from numba.np.arrayobj import numpy_transpose
from scipy.spatial.distance import num_obs_y
import h5py


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
        self.active_function = lambda x: scipy.special.expit(x)

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
        self.weigh_ho += self.lr * numpy.dot((output_loss * final_outputs * (1 - final_outputs)), numpy_transpose(hidden_outputs))
        self.weigh_ih += self.lr * numpy.dot((hidden_loss * hidden_outputs * (1 - hidden_outputs)), numpy_transpose(inputs))

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

def load_train_date():

    for data_index in range(6):
        data_address = "D:\DATA\机器学习tyy_sb_need\chendoudata/sample_feature_{}.mat".format(data_index)
        data = h5py.File(data_address)
        training_date_list = data["date{}".format(data_index)]
        return training_date_list



input_nodes=784
hide_nodes=100
out_nodes=10
learning_rate=0.3

n=NN(input_nodes,hide_nodes,out_nodes,learning_rate)

# 加载训练数据的csv文件
training_date_file = open("文件地址", "r")
training_date_list = training_date_file.readlines()
training_date_file.close()

# 训练
# 遍历所有值
for record in training_date_list:
    all_values = record.split(',')# 以逗号为界分割csv数据
    inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01 # 输入等于除开第一个值的all_values向量除255乘0.99再加0.01
    targets = numpy.zeros(out_nodes)+0.01   # 构建目标空向量
    targets[int(all_values[0])] = 0.99      # int(all_values[0]将第一个文本值变为int数，再将该数对应的向量位赋值0.99
    n.train(inputs, targets)                # 开始训练
    pass



# 测试
# 同上，调用函数query



