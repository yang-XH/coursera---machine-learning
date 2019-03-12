import pandas as pd
import numpy as np
import mat4py
import matplotlib.pyplot as plt


def displayData(X, example_width=20):
    '''
    # 是否传入了example_width值，若没有，则设定为20
    if not example_width in vars():
        example_width = np.round(np.sqrt(X.shape[1]))
    '''
    # Compute rows, cols
    (m, n) = X.shape
    # python3的/是真正意义的除法，结果为float类型，作为高度应转换为Int类型（后面还会作为array的长度）
    # 用//表示整数除法
    example_height = (n // example_width)

    # Compute number of items to display
    # 若显示的图像个数不是平方数，则row取floor，col取ceil(行少列多），最后一行不满
    # 此处应，若能整除，则，若不能，则行少列多？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    # 判断平方后是否整数：与取整后的值是否相同
    if np.sqrt(m) == int(np.sqrt(m)):
        display_rows = int(np.sqrt(m))
        display_cols = m // display_rows
    else:
        display_rows = int(np.floor(np.sqrt(m)))
        # 用//表示整数除法
        display_cols = m // display_rows + 1

    # Between images padding
    pad = 1

    # Setup blank display，黑色应为0？-1？此处相当于设置各图片间隔pad的黑色线条（全黑）
    # 横pad数=rows+1   纵pad数=cols+1
    '''
    display_array = - np.ones(pad + display_rows * (example_height + pad),
                              pad + display_cols * (example_width + pad))
    '''


    
    # 便于后面利用np.hstack（stack的是已经加上右、下pad的x)，此处初始化，最后记得删
    x_stack = np.ones((example_height + pad, 1))
    for j in range(X.shape[0]):
        # TODO ：为啥这里需要转置，哪里做错了吗？？？？？？？？？？？？

        x = X[j, :].reshape(example_height, example_width).T
        # 为每个方块图像的右、下加上pad
        # 报错：'CClass' object is not callable -->np.c_不是函数，应为np.c_[]，而不是np.c_()
        x = np.c_[x, -np.ones((example_height, 1))]
        x = np.r_[x, -np.ones((1, example_width + pad))]
        # hstack() takes 1 positional argument
        x_stack = np.hstack((x_stack, x))
        
    # 删除初始化时增加的第一列1
    x_stack = np.delete(x_stack, 0, axis=1)
    # 对所有小图块进行重定位
    width_sum = display_cols * (example_width + pad)
    x_end = np.ones((1, width_sum))
    for i in range(display_rows):
        print(x_stack[:, width_sum * i: width_sum * (i + 1)].shape)
        # 切片是算前不算后，故stop的数值需注意
        x_end = np.vstack((x_end, x_stack[:, width_sum * i: width_sum * (i + 1)]))

    x_end = np.delete(x_end, 0, axis=0)
    # 给整个图像加上左、上间隔线
    height = display_rows * (example_height + pad)
    width = display_cols * (example_width + pad) + pad
    x_end = np.c_[-np.ones((height, 1)), x_end]
    x_end = np.r_[-np.ones((1, width)), x_end]
    '''reshape的方式不对，
       此方法是：先对每一行进行reshape，是每个小图块的shape，再把所有小图块横向拼接，最后再reshape所有小图块
       然而不能对小图块进行reshape，仍是对图块中的data进行reshape，故与预期不同
    x_stack = x_stack.reshape((display_rows * (example_height + pad), display_cols * (example_width + pad)))
    # 增加左、上的分割线pad
    x_stack = np.c_[-np.ones((display_rows * (example_height + pad), 1))]
    x_stack = np.r_[-np.ones((pad + display_cols * (example_width + pad), 1))]
    print(x_stack.shape)
    assert x_stack.shape == (pad + display_cols * (example_width + pad), display_rows * (example_height + pad) + pad)
    '''
    plt.imshow(x_end, cmap=plt.cm.gray)
    plt.show()








'''
Setup the parameters you will use for this part of the exercise
   20x20 Input Images of Digits
   10 labels, from 1 to 10
   (note that we have mapped "0" to label 10)
'''
input_layer_size  = 400
num_labels = 10
'''
=========== Part 1: Loading and Visualizing Data =============
  We start the exercise by first loading and visualizing the dataset.
  You will be working with a dataset that contains handwritten digits.
'''

# Load Training Data
print('Loading and Visualizing Data ...')
# loadmat的data是dict形式
data = mat4py.loadmat('ex3data1.mat')
'''
   20x20 Input Images of Digits
   10 labels, from 1 to 10
'''
X = np.array(data['X'])  # 5000×400
y = np.array(data['y'])  # 5000×1
print(y.shape)
print(X.shape)
m = X.shape[0]

# Randomly select 100 data points to display
# np.random.randint生成[, m),size大小的随机整数
rand_indices = np.random.randint(m, size=100)  # 或将0~5000随机排序，取前100个
sel = X[rand_indices, :]
displayData(sel)