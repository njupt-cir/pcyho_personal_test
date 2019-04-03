# coding:utf-8

import numpy as np
import tensorflow as tf

inputX = np.random.rand(100)
inputY = np.multiply(3, inputX) + 1
x = tf.placeholder('float32')
weight = tf.Variable(0.25)
blas = tf.Variable(0.25)
y = tf.multiply(weight, x) + blas
y_ = tf.placeholder('float32')
loss = tf.reduce_sum(tf.pow((y - y_), 2))
trainstep = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for _ in range(1000):
        sess.run(trainstep, feed_dict={x: inputX, y_: inputY})
        if _ % 20 == 0:
            print('w is: ', weight.eval(session=sess), '\n', 'biass is: ', blas.eval(session=sess))
