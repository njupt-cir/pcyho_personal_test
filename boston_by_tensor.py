# encoding:utf-8

import numpy as np
from sklearn import datasets
import tensorflow as tf
import matplotlib.pyplot as plt

boston = datasets.load_boston()
x1 = np.random.randint(50, 100, 100)
y1 = 1.7 * x1 + 12 + np.random.randn(1)

inX = tf.placeholder(tf.float32)
ouY = tf.placeholder(tf.float32)
w = tf.Variable(0.1)
b = tf.Variable(0.1)

y_ = tf.multiply(w, inX) + b
cost = tf.reduce_sum(tf.pow((y1 - y_), 2))
trainstep = tf.train.GradientDescentOptimizer(0.02).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for _ in range(100):
        sess.run(trainstep, feed_dict={inX: x1, ouY: y1})
    fig, ax = plt.subplots()
    ax.scatter(x1, y1, edgecolors=(0, 0, 0))
    ax.plot(x1, w.eval(session=sess) * x1 + b.eval(session=sess), 'k--')
    print(w.eval(session=sess), ' \n', b.eval(session=sess))
    plt.show()

"""
CRIM按城镇人均犯罪率
ZN占住宅用地的比例超过25,000平方英尺。
INDUS每个城镇非零售业务占比的比例
CHAS Charles River虚拟变量（如果管道限制河流则= 1;否则为0）
NOX一氧化氮浓度（每千万份）
RM每间住宅的平均房间数
AGE在1940年之前建造的自住单位比例
DIS加权距离到波士顿的五个就业中心
RAD径向高速公路可达性指数
每10,000美元的税收全价物业税率
PTRATIO城镇的师生比例
B 1000（Bk - 0.63）^ 2其中Bk是城镇黑人的比例
LSTAT％人口状况较低
MEDV自住房屋的中位数价值1000美元
"""
