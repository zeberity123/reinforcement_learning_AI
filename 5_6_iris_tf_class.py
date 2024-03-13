# 5_6_iris_tf_class.py
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

tf.disable_v2_behavior()


# 퀴즈
# iris 모델의 텐서플로 버전을 클래스 버전으로 업그레이드 하세요
def make_xy():
    iris = pd.read_csv('data/iris.csv')
    # print(iris)

    enc = preprocessing.LabelBinarizer()        # onehot
    y = enc.fit_transform(iris.variety)

    return np.float32(iris.values[:, :-1]), y


class Dense:
    def __init__(self, n_class, has_relu):
        self.n_class = n_class
        self.has_relu = has_relu

    def __call__(self, *args, **kwargs):
        w = tf.Variable(tf.random_uniform([kwargs['x'].shape[1], self.n_class]))
        b = tf.Variable(tf.zeros([self.n_class]))

        z = tf.matmul(kwargs['x'], w) + b

        if not self.has_relu:
            return z

        return tf.nn.relu(z)


class Sequential():
    def __init__(self, input_shape, output_shape, layers):
        self.x = tf.placeholder(tf.float32, shape=input_shape)
        self.y = tf.placeholder(tf.float32, shape=output_shape)

        output = self.x
        for layer in layers:
            output = layer(x=output)

        self.hx = output

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def predict(self, x):
        return self.sess.run(self.hx, {self.x: x})

    def fit(self, x, y):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.hx, labels=y)
        train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)  # SGD

        self.sess.run(train, {self.x: x, self.y: y})


def tf_model():
    x, y = make_xy()
    # x = preprocessing.scale(x)

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    model = Sequential(input_shape=[None, 4], output_shape=[None, 3], layers=[
        Dense(n_class=12, has_relu=True),
        Dense(n_class=3, has_relu=False),
    ])

    for i in range(1000):
        model.fit(x_train, y_train)
        p = model.predict(x_test)

        p_arg = np.argmax(p, axis=1)
        y_arg = np.argmax(y_test, axis=1)

        if i % 10 == 0:
            print(i, ':', np.mean(p_arg == y_arg))


tf_model()
