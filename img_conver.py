# coding:utf-8
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# 模型参数
learning_rate = 1e-3
training_iters = 200
batch_size = 50
display_step = 5
n_features = 32 * 32 * 3
n_classes = 10
n_fc1 = 384
n_fc2 = 192


def unpickle(file):
    """
    :param file:filepath
    :return: dict
    """
    with open(file, 'rb') as fo:
        dicts = pickle.load(fo, encoding='bytes')
    return dicts


def one_hot(labels):
    """
    :param labels: data labels
    :return: onehot labels
    """
    n_sample = len(labels)
    n_class = max(labels) + 1
    one_hot_labels = np.zeros((n_sample, n_class))
    one_hot_labels[np.arange(n_sample), labels] = 1
    return one_hot_labels


file1 = "E:\\Program\\Data\\cifar-10-python\\cifar-10-batches-py\\data_batch_1"
file2 = "E:\\Program\\Data\\cifar-10-python\\cifar-10-batches-py\\data_batch_2"
file3 = "E:\\Program\\Data\\cifar-10-python\\cifar-10-batches-py\\data_batch_3"
file4 = "E:\\Program\\Data\\cifar-10-python\\cifar-10-batches-py\\data_batch_4"
file5 = "E:\\Program\\Data\\cifar-10-python\\cifar-10-batches-py\\data_batch_5"
test_file = "E:\\Program\\Data\\cifar-10-python\\cifar-10-batches-py\\test_batch"
data1 = unpickle(file1)
data2 = unpickle(file2)
data3 = unpickle(file3)
data4 = unpickle(file4)
data5 = unpickle(file5)
# 训练样本集
x_train = np.concatenate((data1[b'data'], data2[b'data'], data3[b'data'], data4[b'data'], data5[b'data']), axis=0)
y_train = np.concatenate((data1[b'labels'], data2[b'labels'], data3[b'labels'], data4[b'labels'], data5[b'labels']),
                         axis=0)
for _ in x_train:
    _ = tf.reshape(x_train, [-1, 32, 32, 3])
y_train = one_hot(y_train)
# 测试样本集
test = unpickle(test_file)
x_test = test[b'data'][:5000, :]
y_test = one_hot(test[b'labels'])[:5000, :]


# 绘制参数变化
def variable_summaries(var):
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)


# 构建模型
# 输入层
x = tf.placeholder(tf.float32, [None, n_features], name='x-input')
y = tf.placeholder(tf.float32, [None, n_classes], name='y-input')

# 保存图像信息
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 32, 32, 3])
    tf.summary.image('input', image_shaped_input, 10)

# 中间层构建
w_cov = {
    'conv1': tf.Variable(tf.truncated_normal([5, 5, 3, 2], stddev=0.0001), name='w_conv1'),
    'conv2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.0001), name='w_conv2'),
    'fc1': tf.Variable(tf.truncated_normal([8 * 8 * 64, n_fc1], stddev=0.1), name='w_fc2'),
    'fc2': tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.1), name='w_fc2'),
    'fc3': tf.Variable(tf.truncated_normal([n_fc2, n_classes], stddev=0.1), name='w_fc3')
}
b_conv = {
    'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[32]), name='b_conv1'),
    'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[64]), name='b_conv2'),
    'fc1': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc1]), name='b_fc1'),
    'fc2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc2]), name='b_fc2'),
    'fc3': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_classes]), name='b_fc3')
}
x_image = tf.reshape(x, [-1, 32, 32, 3])

# 卷积层1
conv1 = tf.nn.conv2d(x_image, w_cov['conv1'], strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
conv1 = tf.nn.relu(conv1, name='conv1')

# 池化层1
pool1 = tf.nn.avg_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

# LRN层
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='LRN-norm1')

# 卷积层2
conv2 = tf.nn.conv2d(norm1, w_cov['conv2'], strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
conv2 = tf.nn.relu(conv2, name='conv2')

# LRN层
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='LRN-norm2')

# 池化层2
pool2 = tf.nn.avg_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
reshape = tf.reshape(pool2, [-1, 8 * 8 * 64], name='pool2')

# 全连接层1
fc1 = tf.add(tf.matmul(reshape, w_cov['fc1']), b_conv['fc2'])
fc1 = tf.nn.relu(fc1, name='fc1')

# 全连接层2
fc2 = tf.add(tf.matmul(fc1, w_cov['fc2']), b_conv['fc2'])
fc2 = tf.nn.relu(fc2, name='fc2')

# 全连接层3
fc3 = tf.nn.softmax(tf.add(tf.matmul(fc2, w_cov['fc3']), b_conv['fc3']))

# 定义损失
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc3, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
tf.summary.scalar('loss', loss)

# 模型评估
correct_pred = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)
init = tf.global_variables_initializer()

# 会话
with tf.Session() as sess:
    sess.run(init)
    c = []
    total_batch = int(x_train.shape[0] / batch_size)
    # 记录时间
    start_time = time.time()
    for i in range(200):
        for batch in range(total_batch):
            batch_x = x_train[batch * batch_size:(batch + 1) * batch_size, :]
            batch_y = y_train[batch * batch_size:(batch + 1) * batch_size, :]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        print(acc)
        c.append(acc)
        end_time = time.time()
        print('time: ', (end_time - start_time))
    print('Optimization Finished'.center(50))

    # Test
    test_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
    print('testing accuracy: ', test_acc)
    plt.plot(c)
    plt.xlabel('Iter')
    plt.ylabel('Cost')
    plt.title('lr=%f ,ti=%d bs=%d ,acc=%f' % (learning_rate, training_iters, batch_size, test_acc))
    plt.tight_layout()

# 日志信息保存
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/img_conver', sess.graph)
