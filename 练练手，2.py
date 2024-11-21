# 函数
# 形式
# def fun():
#     print('好好学习，天天向上')
# #调用
# fun()
#
#
#
# def printInfo(name,height,weight,hobby):#形式参数：定义时候不占内存。意义上的参数
# #函数代码块
#      print('%s的身高是%fm'%(name,height))
#      print('%s的体重是%fkg'% (name, weight))
#      print('%s的爱好是%s'% (name, hobby))
# printInfo('小名',1.75,80,"唱歌")#实际参数：占用内存地址
# printInfo('小美',1.65,60,"唱歌")
from prompt_toolkit.key_binding.bindings.named_commands import self_insert


# b.参数分类
# 必选参数：定义函数时，形参不赋值，若实参空缺则不能运行。
# 默认参数：定义函数，形参赋值，若实参空缺仍能运行。
# 可选参数：参数个数不确定(*arge)，元组类型。
# 关键字可变参数：参数是字典类型，字典键、key值是字符串(**arges)


# 元组参数
# def sum(*arge):
#       result=0
#       for items in arge:#注意arge是参数，sum只是个函数名
#           result+=items
#           pass
#       print('累加结果是%d'%result)
# sum(1,2,3,4,5)#参数为元组形式


# 字典参数
# def keyfunc (**kwargs):
#       print(kwargs)
#       pass
# dictA={"name":"小明","age":35}
# keyfunc(**dictA)#参数字典形式写法
# keyfunc(name="小美",age=26)#第二种写法

# 结合
# def complexfunc(*arge,**arges):#可选参数必须放到关键字可选参数之前
#       print(arge,arges)
#       pass
# complexfunc(1,2,3)
# complexfunc(name='小美')
# complexfunc(1,2,3,name='小美')


# 函数嵌套
# 定义fun2函数时，fun1函数也参与运行，即在fun2函数中调用fun1，将fun1嵌套在fun2中

# def sum(*arge):
#     result=0
#     for arge in arge:
#         result+=arge
#     print(result)
#     return result
#
# sum(1,1,13,3)

#
# def Sum(*arge):
#     result = 0
#     for i in arge:
#         result += i
#     return result
# print(Sum(1, 2, 3, 4))


# # 列表参数
# def listfun(a):# 如何使用列表作为参数
#     n = len(a)
#     m = a[0:n:2]
#     return m
#
# a=[1,2,3,34,45,5]
# print(listfun(a))
#

# def listfun(a):  # 如何使用列表作为参数
#     n = len(a)  # 获取列表长度
#     m = a[0:n:2]
#     return m
#
#
# print(listfun([1, 2, 3, 3, 4, 5, 7]))

# #传统方法求和
# def fun(x,y):
#     return x+y
# print(fun(1,2))
# #匿名函数求两数中最大值
# n=lambda x,y:x if x>y else y #无函数名,代替传统双分支写法
# print(n (2,3))
# #匿名函数求和
# m=lambda x,y:x+y #无函数名
# print(m (2,3))
# #匿名函数求和
# k=(lambda x,y:x+y)(3,4) #无函数名，直接调用，在该行完成赋值
# print(k)

#
# print(abs(-3))
# print(round(3.499,1))  #3.499保留一位小数
# print(pow(3,3))
# print(3**3)
# print(divmod(7,2))
# print(max([1,2,3,4,5])) #返回给定参数最大值，参数可以是序列
# print(max(1,2,3,4,5 ))
# print(sum(range(10)))
# print(sum(range(10),3)) #最后结果+3
# a,b,c=1,2,3
# print('动态执行的函数={}'.format(eval('a*b+c-30')))
# print(eval('a+b+c',{'c':3,'a':1,'b':3}))
# def TestFun():
#     print('执行函数')
#     pass
# eval('TestFun()')
