# coding:utf-8

import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455

rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
Y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X]

x = tf.placeholder('float32', shape=(None, 2))
