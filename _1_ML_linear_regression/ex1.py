import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#TODO
# 最后的图不太对，点与pdf对应不上，倾斜角度不对

def compute_cost(X_, y_, theta_):
    """
    Compute cost for linear regression
    :param X_: feature
    :param y_: actual output
    :param theta_: parameter
    :return: the cost of using theta as the parameter
             for linear regression to fit the data points in X and y
    """
    m_ = len(y_)
    # 数组array乘法默认的是点乘，而矩阵matrix的乘法默认是矩阵乘法
    cost = 1 / 2 / m_ * np.sum((np.dot(X_, theta_) - y_) ** 2)
    return cost


def gradientDescent(X_, y_, theta_, alpha_, iterations_):
    m_ = len(y_)
    J_history = []
    for i in range(iterations_) :
        # 注意矩阵乘法np.dot()
        theta_ = theta_ - alpha_ / m_ * np.dot(X_.T, (np.dot(X_, theta_) - y_))
        # 若添加元素（e.g. append），不需在定义时确定array,list,Series等的长度
        # 但若修改其中的元素（e.g.特定位置的元素），需要在定义时确定长度
        J_history.append(compute_cost(X_, y_, theta_))
    return theta_, J_history




data = pd.read_table('ex1data1.txt', header=None, sep=',')
m = len(data)
# plt.scatter绘制散点图，横轴数据，纵轴数据
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], color='r')
# 需要show函数，才能显示图形
plt.show()

# 将DataFrame或Series类型转换为array类型
X = np.array(data.iloc[:, 0])
# 在X的第一列添加一列1
X = np.c_[np.ones(len(X)), X]
'''
y.shape = (97,)表示y是一维数组，相当于行向量！！！
对一维数组np.array直接转置没有变化
方法：
    ①赋值时，两个方括号，直接定义为二维数组，
      e.g. np.array([[1,2,3]])为二维数组，shape = (1,3)可直接转置，
           np.array([1,2,3])为一维数组，shape = （3，），转置没有变化
      此时， 
      .T  
      .transpose()都可
    ②通过reshape方法
      a.reshape(-1,1)，-1表示随其他轴个数变化而自动确定
'''
y = np.array([data.iloc[:, 1]])
y = y.T
# 此处np.zeros的参数是（2，1）
theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01
print('\nTesting the cost function')
# compute and display initial cost
J = compute_cost(X, y, theta)
# %字符：标记转换说明符的开始
print('With theta = [0 ; 0]\nCost computed = %f' % J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = compute_cost(X, y, np.array([[-1, 2]]).T)
print('\nWith theta = [[-1],[2]]\nCost computed = %f' % J)
print('Expected cost value (approx) 54.24\n')

print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta, _ = gradientDescent(X, y, theta, alpha, iterations)
# print theta to screen
print('Theta found by gradient descent:\n')
print(theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit
# 散点图与折线图显示在一张图上（一个show()）
# 第一列是X0=1,故只plot第二列
plt.plot(X[:, 1], np.dot(X, theta), color='r')
plt.scatter(X[:, 1], y, color='b')
plt.xlabel('Training data')
plt.ylabel('Linear regression')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]), theta)
# 注意：predict1*3必须用括号括住，否则*此处表示重复作用，会打印此语句及predict值10000次
print('For population = 35,000, we predict a profit of %f\n' % (predict1*10000))
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of %f\n' % (predict2*10000))

print('Visualizing J(theta_0, theta_1) ...\n')
# Grid over which we will calculate J
# np.linspace-样本数量   arange-步长
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
# np.meshgrid()将两个array生成对应的网格点
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
# 注意，此处theta0_vals, theta1_vals已经是网格点形式，按先x轴递增，后y轴递增顺序


def mesh_cost(theta0, theta1):
    # initialize J_vals to a matrix of 0's
    # 接下来对i,j位置赋值，故需定义array大小，也可用np.empty()函数
    # np.zeros()的参数是一个tuple，表示多维数组的大小
    Jvals = np.zeros((len(theta0[1]), len(theta1[0])))
    for i in range(len(theta0[1])):
        for j in range(len(theta1[0])):
            # 注意方括号，t是（2，1）的array
            t = np.array([[theta0[0][i]], [theta1[j][0]]])
            # Fill out J_vals
            Jvals[i, j] = compute_cost(X, y, t)
    return Jvals


# 此处用function建立J_vals与网格点的关系，似乎与直接将J_vals建立为二维数组结果相同
J_vals = mesh_cost(theta0_vals, theta1_vals)
# mpl_toolkits.mplot3d的Axes3D绘制surface图

# creat a new figure
fig = plt.figure()
# 将figure变为3D
ax = Axes3D(fig)
# rstride:行之间的跨度  cstride:列之间的跨度
# rcount:设置间隔个数，默认50个，ccount:列的间隔个数  不能与上面两个参数同时出现
# cmap是颜色映射表
ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1, cmap=plt.cm.jet)
# 绘制从3D曲面到底部的投影,zdir 可选 'z'|'x'|'y'| 分别表示投影到z,x,y平面
# zdir = 'z', offset = -2 表示投影到z = -2上
ax.contour(theta0_vals, theta1_vals, J_vals, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
# 设置z轴的维度，x,y类似
# ax.set_zlim(-2, 2)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()

#plt.figure()
# 数字是等高线的条数
plt.contour(theta0_vals, theta1_vals, J_vals, 28)
# 绘制使cost最小的theta点
plt.scatter(theta[0], theta[1])
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()
