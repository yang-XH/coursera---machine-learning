import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from sklearn import preprocessing


def plotData(X_, y_, x_label, y_label, legend1, legend2):
    """
    由于绘制decision boundary时，需与plotdata绘制于一张图，故此函数中没有plt.show()
    :param X_: np.matrix，不包括全1列
    :param y_: np.matrix，需转换为列向量
    :return:
    """
    # 将y=1的点，与y=0点（x为横纵坐标）分开标记画图
    # 若用plt.scatter散点图，报错：Masked arrays must be 1-D
    # numpy的matrix都为2维，若用scatter，可用to_list()函数转换
    y_ = y_.reshape(-1, 1)
    not_admitted = plt.plot(X_[:, 0][y_ == 0], X_[:, 1][y_ == 0], 'o', color='y')
    admitted = plt.plot(X_[:, 0][y_ == 1], X_[:, 1][y_ == 1], '+', color='b')
    # 此处若是不写plt.legend，则不会显示标签(label标签）
    # 图例及位置：loc函数为位置 ncol为标签有几列
    # handles是被标示的对象，labels是标示内容
    l1 = plt.legend(not_admitted, legend1, loc=3, ncol=1)
    plt.legend(admitted, legend2, loc=1, ncol=1)
    # add l1 as a separate artist to the axes
    # artist（lines, patches）
    plt.gca().add_artist(l1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    return


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def costFunction_logistic(X, y, theta):
    num = len(y)
    J = 0
    # array: ①np.dot点乘（矩阵乘法） ② * 对应相乘
    # matrix:①np.multiply 对应相乘   ② * 矩阵乘法
    # log(1-sigmoid)是翻转sigmoid，使离0越近，cost越小
    J = - 1 / num * np.sum(np.multiply(y, np.log(sigmoid(X * theta)))
                           + np.multiply((1 - y), np.log(1 - sigmoid(X * theta))))
    return J


def gradient(X, y, theta_matrix):
    num = len(y)
    grad = np.zeros(theta_matrix.shape)
    grad = 1 / num * X.T * (sigmoid(X * theta_matrix) - y)
    return grad


def gradient_array(theta_array, X, y):
    # 作为传入minizine函数的参数，此gradient函数的参数theta以及返回值grad都需要是一维数组array
    theta_array = np.mat(theta_array).reshape(-1, 1)
    num = len(y)
    grad = np.mat(np.zeros(theta_array.shape))
    grad = 1 / num * X.T * (sigmoid(X * theta_array) - y)
    return np.array(grad).flatten()


def cost_function_logistic_array(theta_array, X, y):
    # 此时传入的参数顺序theta,X,y
    # 且theta是一维数组
    num = len(y)
    J = 0
    theta = np.mat(theta_array).reshape(-1, 1)
    # array: ①np.dot点乘（矩阵乘法） ② * 对应相乘
    # matrix:①np.multiply 对应相乘   ② * 矩阵乘法
    # log(1-sigmoid)是翻转sigmoid，使离0越近，cost越小
    J = - 1 / num * np.sum(np.multiply(y, np.log(sigmoid(X * theta))) + np.multiply((1 - y), np.log(1 - sigmoid(X * theta))))
    J = np.array(J).flatten()
    return J


def plotDecisionBoundary(theta, X, y, x_label, y_label, legend1, legend2):
    """
    decision boundary是g(θTx)=0.5时，即θTx=θ0+θ1x1+θ2x2=0时
    当feature数量(不包括全1的列) <=2，decision boundary是一天直线，只需两个点确定
    :param theta: np.array，一维数组，(array与list计算时，*表示对应相乘)
    :param X: np.matrix，包括全1列；若feature数<=3，则为原始数据，若feature数>3，则为polynomial形式
    :param y:
    :return:
    """
    theta = theta.reshape(-1, 1)
    if X[1].size <= 3:
        # 取第一个feature：X1的两个点
        x_axis = np.array([X[:, 1].min(), X[:, 1].max()])
        # θTx=θ0+θ1x1+θ2x2=0移项得到X2，作为横纵坐标（X0全1不算）
        # array与List计算是，*表示对应相乘，但array中的一个数与list*时，表示重复此List，故使x_axis type为array
        y_axis = - 1 / theta[2] * (theta[0] + theta[1] * x_axis)
        plt.plot(x_axis, y_axis)
        # 传入plotData中的X，没有全1列
        plotData(X[:, 1:], y, x_label, y_label, legend1, legend2)
        plt.show()
    else:
        # TODO
        pass  # 此处，当feature数>3时，没太懂怎么体现decision boundary???????????????????????????????????
        # 此处考虑的>3个feature的情况，是feature是由2个初始特征组成的polynomial形式，即，y仍是由x1,x2体现，而不是更多的feature
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        # np.meshgrid()将两个array生成对应的网格点
        # 注意，此处theta0_vals, theta1_vals已经是网格点形式，按先x轴递增，后y轴递增顺序
        # e.g. u = [ [ -1,0,1,2,3] --每个行向量的x轴取值
        #            [ -1,0,1,2,3]]
        #      v = [ [ 1,1,1,1,1]  --每个行向量的y轴取值（高度）
        #            [ 2,2,2,2,2]]
        a, b = np.meshgrid(u, v)
        #??????????????????????//
        # 需将u,v变为2D array的两列，作为fit_transform的输入
        # vstack() takes 1 positional argument，传入tuple
        #uv = np.vstack((u, v))
        #uv = uv.transpose()
        poly = preprocessing.PolynomialFeatures(6)
        #？？？？？？？？？？？？？？？？？？//此处应该用meahgrid后的数据做mapfeature，还是之前的？？？？？？？？？？？？？
        # decision boundary是g(θTx)=0.5时，即θTx=θ0+θ1x1+θ2x2=0时，故此处不需计算sigmoid等
        uv_grid = np.mat([[u_i, v_i] for u_i in u for v_i in v])
        # 用meshgrid，for循环，每次polynomial一个行向量的网格点
        # reshape一共四种写法，要按照上面for的顺序和reshape的顺序写
        # 即，for循环是按v先增大，u再增大，数据从下到上，再从左到右，且z行数=u,列数=v
        z = np.dot(poly.fit_transform(uv_grid), theta).reshape(u.shape[0], v.shape[0]).T
        # 此时X是polynomial形式(第一列为全1，23列为原始数据），而plotdata时，横纵坐标应为X1,X2
        plotData(X[:, 1:3], y, x_label, y_label, legend1, legend2)
        # 然后对z contour，即选取z=0的线
        plt.contour(u, v, z, [0])
        plt.show()
    return


def predict(theta, X):
    """

    :param theta: np.array，1维
    :param X: np.matrix
    :return:
    """
    # matrix只能是2维的，X[0]表示X的第一行数据，而不是第0个维度
    m = len(X[:, 0])
    p = np.zeros((m, 1))
    pos = sigmoid(X * np.mat(theta).reshape(-1, 1)) > 0.5
    p[pos] = 1
    return p


if __name__ == '__main__':
    # pd.read_csv默认分隔符是逗号‘，’，而pd.read_table默认分隔符是\t
    # 数据是以','分隔的，故read_csv直接读，若用read_table，需设定sep=','

    data = pd.read_csv('ex2data1.txt', header=None)
    X = np.mat(data.iloc[:, :2])
    # y转换为matrix是行向量，需转换为列向量
    y = np.mat(data.iloc[:, 2]).reshape(-1, 1)
    print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
    # TODO
    # 若legend不同[]括起来，则每个点对应legend中的一个字母？？？？？？？？？？？？？？？？？？？？？？？
    legend1 = ['Not admitted']
    legend2 = ['Admitted']
    x_label = 'Exam 1 score'
    y_label = 'Exam 2 score'
    plotData(X, y, x_label, y_label, legend1, legend2)
    plt.show()

    # matrix.size是所有数据个数，shape是各个维度的个数
    m, n = X.shape
    X = np.c_[np.ones(m), X]
    # np.zeros接收tuple
    initial_theta = np.mat(np.zeros((n + 1, 1)))
    cost = costFunction_logistic(X, y, initial_theta)
    grad = gradient(X, y, initial_theta)
    print('Cost at initial theta (zeros): %f\n' % cost)
    print('Expected cost (approx): 0.693\n')
    print('Gradient at initial theta (zeros): \n')
    print(grad)
    print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')
    # print(initial_theta.shape)

    # Compute and display cost and gradient with non-zero theta
    test_theta = np.matrix('-24; 0.2; 0.2')
    cost = costFunction_logistic(X, y, test_theta)
    grad = gradient(X, y, test_theta)
    print('Cost at initial theta (zeros): %f\n' % cost)
    print('Expected cost (approx): 0.218\n')
    print('Gradient at test theta (zeros): \n')
    print(grad)
    print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')
    '''
    ============= Part 3: Optimizing using fminunc  =============
      In this exercise, you will use a built-in function (fminunc) to find the
      optimal parameters theta.
      
    scipy库里面的minimize函数来替代matlab里的fminunc
    minimize函数：参数①fun(cost)
                      ②theta
                      ③jac(gradient)
                  3个参数的theta，都需要是shape(n,)的一维数组，不能是matrix，也不是n维array
                  且函数参数的顺序为：先theta，后X，y
    将matrix转换为array后，是n维数组，需要变为1维数组！
    args=(X, y) tuple是除theta外，函数中的其他参数
    '''
    theta_array = np.array(initial_theta).flatten()
    result = op.minimize(fun=cost_function_logistic_array, x0=theta_array,
                         args=(X, y), method='TNC', jac=gradient_array)
    # op.minimize返回的是OptimizeResult object，其中包括多个属性，fun为cost，x为theta(array类型）
    cost = result.fun
    theta = result.x
    print('Cost at theta found by fminunc: %f\n' % cost)
    print('Expected cost (approx): 0.203\n')
    print('theta: \n')
    print(theta)
    print('Expected theta (approx):\n')
    print(' -25.161\n 0.206\n 0.201\n')
    plotDecisionBoundary(theta, X, y, x_label, y_label, legend1, legend2)


    '''
    ============== Part 4: Predict and Accuracies ==============
      After learning the parameters, you'll like to use it to predict the outcomes
      on unseen data. In this part, you will use the logistic regression model
      to predict the probability that a student with score 45 on exam 1 and 
      score 85 on exam 2 will be admitted.
    
      Furthermore, you will compute the training and test set accuracies of 
      our model.
    
      Your task is to complete the code in predict.m
    
      Predict probability for a student with score 45 on exam 1 
      and score 85 on exam 2 
    '''
    prob = sigmoid(np.matrix('1 45 85') * np.mat(theta).reshape(-1, 1))
    prob = float(prob)
    # python的print是默认换行的
    print('For a student with scores 45 and 85, we predict an admission probability of %f' % prob)
    print('Expected value: 0.775 +/- 0.002\n\n')
    # Compute accuracy on our training set
    p = predict(theta, X)
    # only size-1 arrays can be converted to Python scalars
    # p ==y 是m行1列的array，应用np.mean需要1行m列，flatten()函数
    print('Train Accuracy: %f' % (np.mean((p == y).flatten()) * 100))
    print('Expected accuracy (approx): 89.0')


