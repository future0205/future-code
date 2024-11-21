#print函数用法
# name = ("小明")
# height = ("180cm")
# print(name+"的身高"+ height)
#
# name = ('小名')
# height = 180
# print("%s身高是%6.2fcm"%(name, height))
#
#
#
# print('11在字符串的类型是%s'%type(11))格式化
# print('11.2在字符串的类型是%s'%type(11.2))
# print(f'字符串在python中的类型是{type("字符串")}')快速格式化
from anaconda_project.internal.conda_api import result


#
# # #定义变量存储布尔类型数据
# # bool_1=True
# # bool_2=False
# # print(f"bool_1变量内容是{bool_1},类型是：{type(bool_1)}")
# #比较运算符得到布尔类型
# num1=10
# num2=10
# x = num1!=num2
# print(x)
#
#



#input函数输入
# age=input("请输入你年龄：")
# age=int(age)
# if age>18:
#     print("已经成年")
#     pass






# if elif else
# import random
#
# n = random.randint(1, 10)
# x = int(input("猜一下这个数是多少:"))
# if x > n:
#     print("数偏大了")
#     x = int(input("再猜一次"))
#     if x > n:
#         print("又偏大了")
#         x = int(input("最后猜一次"))
#         if x != n:
#             print("你输了,这个数为%s" %n)
#         else:
#             print("你猜对了")
#     elif x < n:
#         print("偏小了")
#         x = int(input("最后猜一次"))
#         if x != n:
#             print("你输了,这个数为%s" %n)
#         else:
#             print("你猜对了")
#     else:
#         print("你猜对了")
# elif x < n:
#     print("偏小了")
#     x = int(input("再猜一次"))
#     if x > n:
#         print("偏大了")
#         x = int(input("最后猜一次"))
#         if x != n:
#             print("你输了,这个数为%s" %n)
#         else:
#             print("你猜对了")
#     elif x < n:
#         print("又偏小了")
#         x = int(input("最后猜一次"))
#         if x != n:
#             print("你输了,这个数为%s" %n)
#         else:
#             print("你猜对了")
#     else:
#         print("你猜对了")
# else:
#     print("你猜对了")





# 怎么读取文件
# import scipy.io as scio  #导入scipy包
# import numpy as np       # 导入numpy库
# import  matplotlib.pyplot as plt
# ## 定义数据获取函数  ##
# def DataAcquision(FilePath):
#     data = scio.loadmat(FilePath)     # 加载mat数据
#     data_key_list = list(data.keys())  # mat文件为字典类型，获取字典所有的键并转换为list类型
#     accl_key = data_key_list[3]        # 获取'DE_time'
#     accl_data = data[accl_key].flatten()  # 获取'X108_DE_time'所对应的值，即为振动加速度信号,并将二维数组展成一维数组
#     return accl_data
#
# FilePath = r'E:\DATA\德国Paderborn大学数据\K001\N09_M07_F10_K001_1.mat'  #文件夹路径
# accl_data = DataAcquision(FilePath)
# fs = 12000                 # 采样率为12k
# data_len = len(accl_data)  # 获取数据长度
# t = data_len/fs            # 计算采样时间  （采样时间 ）= （数据长度）/（采样率）
# x = np.linspace(0,t,data_len)   # 划分x轴的数据，其长度应为数据长度
# plt.figure(figsize=(20,5))      #设置绘图大小
# plt.xlabel('t', fontsize=18)    #设置x轴标签名字为‘t’，设置其大小为18
# plt.ylabel('acclerationdata', fontsize=18)   #设置y轴标签名字为‘accleration data’，设置其大小为18
# plt.title('12k-DriveEndFault-1730-0.007-InnerRace',fontsize=18)  #设置标题
# plt.plot(x, accl_data)      # 绘图





# while循环
# i=9
# while i >= 1:
#      j=i
#      while j >= 1:
#          print(f"{i}*{j}={i*j}\t",end='')
#          j -= 1
#      print()
#      i -= 1
#      pass
# pass
#
# for i in range(5,9): # range(5,8)生成列表[5,6,7]，对该列表进行循环
#     print(i)
# for j in range(0,6,2): # range(0,6,2)生成列表[0,2,4]，对该列表进行循环
#     print(j)







# 操作终止方法，continue与break
# for x in 'i love you':  # 对字符串'i love you'每个元素进行循环，每次读取到的值赋给x
#     if x=='o':          # 如果x为'o'，跳过本次循环
#         continue
#     elif x=='e':        # 如果x为'o'，结束本次循环
#         break
#     print(x)


# zh1="www"
# pw1='123'
# for i in range(3):
#     zh2=input('请输入账号:')
#     pw2=input('请输入密码:')
#     if zh1==zh2 and pw1==pw2:
#         print("成功")
#         break
#     else:
#         print("输入账号密码错误")
# else:
#     print("三次均失败")










# 字符串操作
# capitalize()：首字母变大写
#
# endswith/startswith()：是否以x结束/开始
#
# find()：检测x是否在字符串中
#
# isalnum()：判断是否是字母和数字
#
# isalpha()：判断是否是字母
#
# isdigit()：判断是否是数字
#
# islower()：判断是否是小写
#
# join()：循环取出所有值用xx去连接
#
# lower/upper()：大小写转换
#
# swapcase()：大写变小写，小写变大写
#
# lstrip/rstrip/strip()：移除左/右/两侧空白
#
# split()：切割字符串
#
# title()：把每个单词的首字母变大写
#
# replace(old，new，count=None)：old为被替换字符串，new为将要替换成的字符串，count表示替换前多少个）
#
# count()：统计出现的总次数



# t='wfofw, 123, 23, 3'
# print(t.find('o'))   # 检测'o'是否在t中
# print(t.find('x'))   # 检测'x'是否在t中
# print(t.endswith('w'))   # 检测是否以'w'结尾
# print(t.capitalize())    # 将首字母变大写
# print(t.split(",",2))    # 分割
# print(t.title())
#










# 列表操作
# append()：在列表后面追加元素
#
# count()：统计元素出现的次数
#
# extend()：扩展，相当于批量添加
#
# index()：获取指定元素索引号
#
# insert()：在指定位置插入
#
# pop()：删除最后一个元素
#
# reverse()：反转列表
#
# remove()：移除元素
#
# sort()：列表排序

# j=[1,34]         # j列表
# i=[1,2,'w',3.4,'reset']     # i列表
# i.extend(j)      # 将j列表添加到i列表后面，是将j每个元素添加到后面
# print(i)
# i.append(j)      # 将j列表添加到i列表后面，是将j整个列表添加到后面
# print(i)
# i.insert(2,j)    # 在i列表的第2个元素后添加j列表，此时是添加整个j列表
# print(i)
# i.reverse()
# print(i)


# i=list(range(0,20))
# i[0]=3 #替换第一个元素
# print(i)
# del i[1] #删除第二个元素
# print(i)
# del i[2:4:2] #通过切片批量删除
# print(i)
# i.remove(15) #移除指定元素
# print(i)
# i.pop() #移除最后元素
# print(i)
# i.pop(1) #指定位置移除元素
# print(i)
# print(i.index(12)) #求目标元素12的索引位置






# 元组，元素不能更改
# i=(1,'we',3,[2,1,1])
# i[3][0]=1   # 将第4个元素中第1个元素赋值为1
# print(i)


# i=tuple(range(10))
# print(i,i.count(0),i.index(0),end=" ") # 打印元组中0的个数，0的索引






# 字典
# 修改元素：通过建找到对应值修改；
# 删除元素：del删除指定的元素‘；
# 新增元素：使用 变量名['键']=数据 添加元素；
# 获取值：values
# 获取键：keys
# 统计个数：len()查看字典键值对个数
# 获取键值对：items
# 删除指定键：pop('键')删除指定键

# i = {"职业": "学生",}
# i['age']='20'#添加新键值对
# # print(i)#输出字典
# # print(len(i)) #键的长度，键值对的个数
# # print(type(i))  #打印类型
# # print(i['age'])#选择键获取值
# # print(i.items())#获取所有键值
# # for item in i.items():#遍历方法不同
# #     print(item)
# # for key,value in i.items():
# #     print('%s==%s'%(key,value))
# print(sorted(i.items(),key=lambda d:d[1], reverse=False))#排序，由于键值对是元组，0代表key,1代表数据
# i['职业']='工程师'#更新
# i.update({'age':22}) #更新或者加添加
# print(i)#输出字典
# del i['age']#关键字del删除指定键
# i.pop('职业')#通过pop删除指定键
# print(i)#输出字典







# 通用操作
# 合并操作+：两个对象相加操作，会合并两个对象(适用于字符串，列表，元组)
# 复制*：对象自身按指定次数复制(适用于字符串，列表，元组)
# 判断元素是否存在in：判断指定元素是否存在与对象(适用于字符串，列表，元组，字典）
# a=('我',2)
# b=(1,)
# c={'你':1}
# print(a+b) # 合并+
# print(a*3) # a列表复制3次
# print('我'in a)
# print(3 in a)
# print('你'in c) # in判断在不在典


