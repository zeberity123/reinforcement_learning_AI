# 1_4_iris.py
import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection


def make_xy():
    iris = pd.read_csv('data/iris.csv')
    # print(iris)

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(iris.variety)

    return np.float32(iris.values[:, :-1]), y


def save_model():
    x, y = make_xy()
    # x = preprocessing.scale(x)

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential([
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=100, verbose=2,
              validation_data=(x_test, y_test))

    model.save('model/iris.keras')


def load_model():
    x, y = make_xy()
    # x = preprocessing.scale(x)

    data = model_selection.train_test_split(x, y, train_size=0.7)
    _, x_test, _, y_test = data

    model = keras.models.load_model('model/iris.keras')
    print('acc :', model.evaluate(x_test, y_test, verbose=0))

    # x = np.array(["5.1,3.5,1.4,.2".split(',')])
    x = np.array([[5.1, 3.5, 1.4, .2]])
    print(x.shape)

    p = model.predict(x, verbose=0)
    print(p)
    p_arg = np.argmax(p, axis=1)
    print('result :', p_arg)
    print('result :', ['setosa', 'versicolor', 'virginica'][p_arg[0]])


# save_model()
load_model()
