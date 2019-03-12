import numpy as np
import pandas as pd
from _2_ML_classification_regularization import ex2
import matplotlib.pyplot as plt
from sklearn import preprocessing
#import sklearn.preprocessing
import scipy.optimize as op

def mapFeature(a, b):
    # TODO : 可补全
    """
    内建函数：sklearn.preprocessing.PolynomialFeatures来进行特征的构造
    MAPFEATURE Feature mapping function to polynomial features

   MAPFEATURE(a, b) maps the two input features
   to quadratic features used in the regularization exercise.

   Returns a new feature array with more features, comprising of
   a, b, a.^2, b.^2, a*b, a*b.^2, etc..

   Inputs a, b must be the same size
    :param a: (m,1)维matrix
    :param b:
    :return:
    """
    degree = 6
    out = np.ones(len(a))
    '''
    % 根据循环，共（1+7）*7/2=28项
for i = 1:degree
    for j = 0:i
        % X1的平方从i到0，X2的平方从0到i
        % 在out的最后一列后面加上新的一列，故第一列仍为1，故i从1开始
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

% 以下为另一种写法，则i从0开始，计算出的值从1开始（而不是从X1,X2开始）
% k = 1;
% for i = 0:degree
%     for j = 0:i
%         out(:,k) = (X1.^(i-j)).*(X2.^j);
%         k = k + 1;
%     end
% end

end
    '''
    return   #


def costFunctionReg(theta_array, X, y, lambda_reg):
    """

    :param theta_array: np.array，1维数组（n，），不是n维数组，便于应用在minimize函数中
    :param X: np.matrix
    :param y: np.matrix
    :param lambda_reg:
    :return:
    """
    num = len(y)
    theta = np.mat(theta_array).reshape(-1, 1)
    J = 0
    # log(1-sigmoid)是翻转sigmoid，使离0越近，cost越小
    # 全1列的X对应的theta，在正则化时，不计算入内
    # matrix只能2维，matrix[a]表示第a行数据，故应是theta[1:].T，而不是theta.T[1:]
    # 因为theta转置后，是行向量，故此时仅有theta[0]，没有theta[1:]
    J = - 1 / num * np.sum(np.multiply(y, np.log(ex2.sigmoid(X * theta)))
                           + np.multiply((1 - y), np.log(1 - ex2.sigmoid(X * theta)))) \
        + lambda_reg / 2 / num * theta[1:].T * theta[1:]
    return J


# 后面会用到minimize函数，需要cost,grad是2个分开的函数
def gradientReg(theta_array, X, y, lambda_reg):
    """

    :param theta_array: np.array，1维数组（n，），不是n维数组，便于应用在minimize函数中
    :param X:
    :param y:
    :param lambda_reg:
    :return:
    """
    num = len(y)
    theta = np.mat(theta_array).reshape(-1, 1)
    grad = np.zeros((theta.size, 1))
    # np.row_stack()的参数是tuple，2对括号
    grad = 1 / num * X.T * (ex2.sigmoid(X * theta) - y) \
           + np.row_stack((0, theta[1:])) * lambda_reg / num
    return grad




data = pd.read_csv('ex2data2.txt', header=None)
# TODO
# dataframe类型为何不能写data[:, :2]????????????????????????????????????????????????????????
# matrix,array都可以？？？？？？？？？？？？？？？？？？？？？？？？？？？
#print(data[0,1]) data[0]是第一列，header=None时,0,1,2....是列的索引
X_ori = data.iloc[:, :2]
X_ori = np.mat(X_ori)
y = data.iloc[:, 2]
y = np.mat(y).reshape(-1, 1)
x_label = 'Microchip Test 1'
y_label = 'Microchip Test 2'
legend1 = ['y = 0']
legend2 = ['y = 1']
ex2.plotData(X_ori, y, x_label, y_label, legend1, legend2)
# ex2.plotData中没有plt.show()
plt.show()

'''
=========== Part 1: Regularized Logistic Regression ============
  In this part, you are given a dataset with data points that are not
  linearly separable. However, you would still like to use logistic
  regression to classify the data points.

  To do so, you introduce more features to use -- in particular, you add
  polynomial features to our data matrix (similar to polynomial
  regression).
'''
'''
 class PolynomialFeatures(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin)
 Generate polynomial and interaction features.
 |
 |  Generate a new feature matrix consisting of all polynomial combinations
 |  of the features with degree less than or equal to the specified degree.
 |  For example, if an input sample is two dimensional and of the form
 |  [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
'''
# 通过preprocessing.PolynomialFeatures实现mapfeature的功能，即feature的polynomial形式
poly = preprocessing.PolynomialFeatures(6)
X = poly.fit_transform(X_ori)
print(type(X))
#a = np.array(X)
#print(a)
# Initialize fitting parameters
# np.zeros(),np.ones()若生成多维的array，在括号中的参数为tuple，即2对括号！！！！！！不要忘记！！！
initial_theta = np.zeros(X[0, :].size)

# Set regularization parameter lambda to 1
lambda_reg = 1
cost = costFunctionReg(initial_theta, X, y, lambda_reg)
grad = gradientReg(initial_theta, X, y, lambda_reg)
print('Cost at initial theta (zeros): %f' % cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only:')
# 保留4位小数
print(np.round(grad[:5], 4))
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

# Compute and display cost and gradient with all-ones theta and lambda = 10
test_theta = np.ones(X[0, :].size)
cost = costFunctionReg(test_theta, X, y, 10)
grad = gradientReg(test_theta, X, y, 10)

print('Cost at test theta (with lambda = 10): %f' % cost)
print('Expected cost (approx): 3.16')
print('Gradient at test theta - first five values only:')
print(grad[:5])
print('Expected gradients (approx) - first five values only:')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

'''
============= Part 2: Regularization and Accuracies =============
  Optional Exercise:
  In this part, you will get to try different values of lambda and
  see how regularization affects the decision coundart

  Try the following values of lambda (0, 1, 10, 100).

  How does the decision boundary change when you vary lambda? How does
  the training set accuracy vary?
'''
initial_theta = np.zeros(X[0, :].size)
lambda_reg = 1
'''
scipy库里面的minimize函数来替代matlab里的fminunc
minimize函数：参数①fun(cost)
                  ②theta
                  ③jac(gradient)
              3个参数的theta，都需要是shape(n,)的一维数组，不能是matrix，也不是n维array
              且函数参数的顺序为：先theta，后X，y
'''
result = op.minimize(fun=costFunctionReg, x0=initial_theta,
                     args=(X, y, lambda_reg), method='TNC', jac=gradientReg)
# op.minimize返回的是OptimizeResult object，其中包括多个属性，fun为cost，x为theta(array类型）
cost = result.fun
theta = result.x
x_label = 'Microchip Test 1'
y_label = 'Microchip Test 2'
legend1 = ['y = 0']
legend2 = ['y = 1']
ex2.plotDecisionBoundary(theta, np.mat(X), y, x_label, y_label, legend1, legend2)
# Compute accuracy on our training set
p = ex2.predict(theta, X)
# only size-1 arrays can be converted to Python scalars
# p是matrix，只能是2维，尽管flatten仍是2维，应转换为array，再.flatten()
print('Train Accuracy: %f' % (np.mean((np.array(p == y)).flatten()) * 100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)')


