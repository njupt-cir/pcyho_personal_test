# coding:utf-8
# 反向传播测试
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
# 随机数生成种子
seed = 22333
# 生成32维2列张量
rng = np.random.RandomState(seed)
X = rng.rand(32, 2)
# 筛选数据
Y = [[int(x0 + x1) < 1] for (x0, x1) in X]

x = tf.placeholder('float32', shape=(None, 2))
y_ = tf.placeholder('float32', shape=(None, 1))
# 生成随机数据
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 利用梯度下降函数进行训练
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step=tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
# 训练
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print('w1: ', sess.run(w1))
    print('w2: ', sess.run(w2), end='\n\n')

    STEPS = 3000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
    print('w1: ', sess.run(w1))
    print('w2: ', sess.run(w2))

"""
w1:  [[-0.8113182   1.4845988   0.06532937]
 [-2.4427042   0.0992484   0.5912243 ]]
w2:  [[-0.8113182 ]
 [ 1.4845988 ]
 [ 0.06532937]]

w1:  [[-0.7186777   0.77121365  0.1069645 ]
 [-2.3335981  -0.1400171   0.58601296]]
w2:  [[ 0.01736024]
 [ 0.7769013 ]
 [-0.07439668]]
 """
