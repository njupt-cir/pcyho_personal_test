# encoding:utf-8
import tensorflow as tf
import numpy as np

inputX = np.random.rand(3000, 1)
noise = np.random.normal(0, 0.05, inputX.shape)
outputY = inputX * 4 + 1 + noise

# first
weight1 = tf.Variable(np.random.rand(inputX.shape[1], 4))
biase1 = tf.Variable(np.random.rand(inputX.shape[1], 4))
x1 = tf.placeholder(tf.float64, [None, 1])
y1_ = tf.matmul(x1, weight1) + biase1

# secound
weight2 = tf.Variable(np.random.rand(4, 1))
biase2 = tf.Variable(np.random.rand(inputX.shape[1], 1))
y2_ = tf.matmul(y1_, weight2) + biase2

y = tf.placeholder(tf.float64, [None, 1])

loss = tf.reduce_mean(tf.reduce_sum(tf.square((y2_ - y)), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.25).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        sess.run(train, feed_dict={x1: inputX, y: outputY})

    print(weight1.eval(sess), end='\n\n')
    print(weight2.eval(sess), end='\n\n')
    print(biase1.eval(sess), end='\n\n')
    print(biase2.eval(sess), end='\n\n')

    x_data = np.array([[1.], [2.], [3.]])
    print(sess.run(y2_, feed_dict={x1: x_data}))

"""
[[1.02313731 1.10970858 1.52845992 1.93483463]]

[[0.50182516]
 [0.55346929]
 [0.73715903]
 [0.90013645]]

[[-0.2827525  -0.29487782 -0.4429092  -0.60995606]]

[[2.18312809]]

[[ 4.9984506 ]
 [ 8.99440945]
 [12.99036829]]

some time may get [nan], just run again
"""
