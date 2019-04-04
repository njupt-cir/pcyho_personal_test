# encoding:utf-8

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# 生成随机数据
X = np.random.randint(100, size=100)
E = np.random.randint(300, size=100)
a = 100
b = 10
Y = a + b * X + E
X = np.reshape(X, newshape=(100, 1))
Y = np.reshape(Y, newshape=(100, 1))
################################################
# 最小二乘法计算回归
X_bar = X.sum() / 100
up = ((X - X_bar) * Y).sum()
down = (X ** 2).sum() - ((X_bar * 100) ** 2) / 100
W = up / down
B = (Y - W * X).sum() / 100
Y_line = W * X + B
#################################################
# 调用库函数回归
lr = LinearRegression()
lr.fit(X, Y)

y_hat = lr.predict(X)
# 画图
plt.figure()

plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X, Y, label='data')
plt.plot(X, Y_line, label='result')
plt.plot(X, y_hat, label='ragiulation_line')
# plt.plot(X, Y_line, X, y_hat, 'b--')
plt.legend()
plt.show()
# save data
df = pd.DataFrame(list(zip(X.T.tolist()[0], Y.T.tolist()[0], y_hat.T.tolist()[0],(y_hat-Y_line).T.tolist()[0])))
df.columns = ['X', 'Y', 'y_hat','Y-y_hat']
df.to_csv('out.csv', encoding='gbk', index=False)
# 读取数据
Data = pd.read_csv('out.csv')
data_X = df.values[:, 0].tolist()
data_Y = df.values[:, 1].tolist()
data_Y_hat = df.values[:, 2].tolist()
# print(data_X)
# print(data_Y)
# print(data_Y_hat)
