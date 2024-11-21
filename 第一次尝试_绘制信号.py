import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fftpack, stats
import random
import os
import statsmodels.tsa.api as smt
import statsmodels


from matplotlib import rcParams

config = {
    "font.family": 'serif', # 衬线字体
    "font.size": 10, # 相当于小四大小
    "font.serif": ['SimSun'], # 宋体
    "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)

fs = 1000   # 采样频率1000Hz
f = 10      # 信号频率 10Hz
n = np.arange(0, 4096)  # 生成4096点的序列
print(n)

t = n/fs    # 生成4096点的时间序列
x = np.sin(2 * np.pi * n/fs * f) + 0.5*np.random.randn(4096)  # 生成正弦信号 sin(2*pi*f*t) + 随机信号
plt.figure(figsize=(12,2))
plt.plot(x)
plt.xlabel('time(s)')
plt.ylabel('Amp(m/s2)')
plt.title('正弦信号+噪声信号图')
plt.show()
print(x.shape)

