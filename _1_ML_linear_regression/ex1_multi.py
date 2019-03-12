import numpy as np
import pandas as pd
# 数据标准化的包
from sklearn import preprocessing
import matplotlib.pyplot as plt
from _1_ML_linear_regression import ex1


#TODO
# 为啥运行或者debug会把ex1再运行一遍？？？？？？？？？？？？？？？？？？？

def gradientDescentMulti(X_, y_, theta_, alpha_, num_iters_):
    theta_, J_history_ = ex1.gradientDescent(X_, y_, theta_, alpha_, num_iters_)
    return theta_, J_history_


data = pd.read_table('ex1data2.txt', header=None, sep=',')
X = np.array(data.iloc[:, :2])
# reshape将y转换为列向量
y = np.array(data.iloc[:, 2]).reshape(-1, 1)
m = len(y)
# Scale features and set them to zero mean
# 多变量时，数标准化!!!   此处采用z-score: （x-μ）/σ
X_scale = preprocessing.scale(X, axis=0)
X_scale = np.c_[np.ones(m), X_scale]

alpha = 0.01
num_iters = 400
theta = np.zeros((3, 1))
theta, J_history = gradientDescentMulti(X_scale, y, theta, alpha, num_iters)
plt.plot(list(range(num_iters)), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

print('Theta computed from gradient descent: \n')
print(theta)

# compute the closed form solution for linear regression using the normal equations
# normal equation θ = (XTX)-1 XTy

# 不使用梯度下降，则不需对变量进行数据标准化
# 此处将Numpy的array转换为matrix，乘法则不需写为np.dot
X_ones = np.c_[np.ones(m), X]
X_n = np.mat(X_ones)
y_n = np.mat(y)
theta_n = (X_n.T * X_n).I * X_n.T * y_n
print('Theta computed from the normal equations: \n')
print(theta)
