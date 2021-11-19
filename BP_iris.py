# 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import radviz

# 初始化参数
def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)
    # 创建连接权矩阵   w1为输入层和隐藏层之间的连接权，w2为隐藏层和输出层之间的连接权
    w1=np.random.randn(n_h,n_x)*0.01
    w2=np.random.randn(n_y,n_h)*0.01
    # 创建偏置矩阵
    b1=np.zeros(shape=(n_h,1))
    b2=np.zeros(shape=(n_y,1))

    # 存储参数
    parameters={'w1':w1,'w2':w2,'b1':b1,'b2':b2}

    return parameters

# 前向传播
def forward_propagation(X,parameters):      # X是喂给输入层的x矩阵
    # 初始化之后的参数装载
    w1=parameters['w1']
    w2=parameters['w2']
    b1=parameters['b1']
    b2=parameters['b2']

    # 计算隐藏层的输出a1
    z1=np.dot(w1,X)+b1  # w1和X做矩阵乘法，计算输入层给隐藏层的总输入值z1
    a1=np.tanh(z1)  # 使用tanh函数做隐藏层的激活函数

    # 计算输出层的输出a2
    z2=np.dot(w2,a1)+b2
    a2=1/(1+np.exp(-z2))    # 使用sigoid函数做输出层的激活函数

    # 存储参数
    parameters_back={'z1':z1,'a1':a1,'z2':z2,'a2':a2}

    return a2, parameters_back

# 交叉熵代价函数
def bp_cost(a2,Y):
    column=Y.shape[1]
    logprobs=np.multiply(np.log(a2),Y)+np.multiply((1-Y),np.log(1-a2))
    cost=-np.sum(logprobs)/column

    return cost

# 反向传播
def backward_propagation(parameters,parameter_back,X,Y):
    column=Y.shape[1]
    w2=parameters['w2']

    a1=parameter_back['a1']
    a2=parameter_back['a2']

# 计算代价函数的偏导数
    dz2=a2-Y      # 输出值a2与Y的偏差
    dw2=(1/column)*np.dot(dz2,a1.T)     # p103(5.11)
    db2=(1/column)*np.sum(dz2,axis=1,keepdims=True)
    dz1=np.multiply(np.dot(w2.T,dz2),1-np.power(a1,2))
    dw1=(1/column)*np.dot(dz1,X.T)
    db1=(1/column)*np.sum(dz1,axis=1,keepdims=True)
    grads={'dw1':dw1,'dw2':dw2,'db1':db1,'db2':db2}

    return grads

# 参数更新
def update_parameters(parameters,grads,learning_rate=0.5):
    w1=parameters['w1']
    w2=parameters['w2']
    b1=parameters['b1']
    b2=parameters['b2']
    dw1=grads['dw1']
    dw2=grads['dw2']
    db1=grads['db1']
    db2=grads['db2']

    # v=v+k   k=-learning_rate * m   m为均方误差对权重求偏导 (p102(5.5))
    w1=w1-dw1*learning_rate
    w2=w2-dw2*learning_rate
    b1=b1-db1*learning_rate
    b2=b2-db2*learning_rate

    parameters={'w1':w1,'w2':w2,'b1':b1,'b2':b2}

    return parameters
#  预测的one-hot编码
def run_model(parameters, x_test, y_test):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    z1 = np.dot(w1, x_test) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))

    # 结果的维度
    n_rows = y_test.shape[0]
    n_cols = y_test.shape[1]

    # 结果存储
    output = np.empty(shape=(n_rows, n_cols), dtype=int)

    for i in range(n_rows):
        for j in range(n_cols):
            if a2[i][j] > 0.5:
                output[i][j] = 1
            else:
                output[i][j] = 0
# 计算准确率

    count = 0
    for k in range(0, n_cols):
        if output[0][k] == y_test[0][k] and output[1][k] == y_test[1][k] and output[2][k] == y_test[2][k]:
            count = count + 1

    acc = count / int(y_test.shape[1]) * 100
    print('准确率：%.2f%%' % acc)

    return output

# 可视化
def result_visualization(x,result):
    cols = y.shape[1]
    type= []  # 类别矩阵

    # 根据数据集的one-hot编码定义类别
    for i in range(cols):
        if result[0][i] == 0 and result[1][i] == 0 and result[2][i] == 1:
            type.append('virginica')
        elif result[0][i] == 0 and result[1][i] == 1 and result[2][i] == 0:
            type.append('versicolor')
        elif result[0][i] == 1 and result[1][i] == 0 and result[2][i] == 0:
            type.append('setosa')

    # 拼接特征和类别矩阵
    real = np.column_stack((x.T, type))

    # 转换成DataFrame类型，添加columns
    df_real = pd.DataFrame(real, index=None, columns=['Sepal Length', 'Sepal Width',
                                                      'Petal Length', 'Petal Width', 'Species'])

    # 将特征列转换为float类型
    df_real[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']] = df_real[[
        'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].astype(float)
    radviz(df_real, 'Species', color=['blue', 'green', 'red', 'yellow'])
    plt.show()

# 建立神经网络
def nn_model(X, Y, n_h, n_input, n_output, num_iterations=50000):
    np.random.seed(3)

    # 初始化参数
    n_x = n_input           # 输入层节点数
    n_y = n_output          # 输出层节点数
    parameters = initialize_parameters(n_x, n_h, n_y)

    # 梯度下降循环
    for i in range(0, num_iterations):
        # 前向传播
        a2, parameters_back = forward_propagation(X, parameters)
        # 代价函数
        cost = bp_cost(a2, Y)
        # 反向传播
        grads = backward_propagation(parameters, parameters_back, X, Y)
        # 更新参数
        parameters = update_parameters(parameters, grads)
        # 每1000次迭代，输出一次代价函数
        if  i % 1000 == 0:
            print('Time=%i，cost=%f' % (i, cost))

    return parameters


if __name__ == "__main__":
    data = pd.read_csv('D:\\pythonProject_iris_bp_classfication\\data_2.csv', header=None)
    x = data.iloc[:, 0:4].values.T
    y = data.iloc[:, 4:].values.T
    parameters = nn_model(x, y, n_h=10, n_input=4, n_output=3, num_iterations=50000)
    result = run_model(parameters, x, y)
    result_visualization(x, result)