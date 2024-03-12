# 4_8_iris_tf.py
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

tf.disable_v2_behavior()


def make_xy():
    iris = pd.read_csv('data/iris.csv')
    # print(iris)

    enc = preprocessing.LabelBinarizer()        # onehot
    y = enc.fit_transform(iris.variety)

    return np.float32(iris.values[:, :-1]), y


def tf_model():
    x, y = make_xy()
    # x = preprocessing.scale(x)

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    x = tf.placeholder(tf.float32, shape=[None, 4])
    y = tf.placeholder(tf.float32, shape=[None, 3])

    w1 = tf.Variable(tf.random_uniform([4, 12]))
    b1 = tf.Variable(tf.zeros([12]))

    w2 = tf.Variable(tf.random_uniform([12, 3]))
    b2 = tf.Variable(tf.zeros([3]))

    # (?, 12) = (?, 4) @ (4, 12)
    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)
    # (?, 3) = (?, 12) @ (12, 3)
    hx = tf.matmul(r1, w2) + b2
    # z2 = tf.matmul(r1, w2) + b2
    # hx = tf.nn.softmax(z2)

    # loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hx)))           # categorical cross entropy
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=hx, labels=y)
    train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)   # SGD

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

        p = sess.run(hx, {x: x_test})

        p_arg = np.argmax(p, axis=1)
        y_arg = np.argmax(y_test, axis=1)

        if i % 10 == 0:
            print(i, ':', np.mean(p_arg == y_arg))

    sess.close()


tf_model()









