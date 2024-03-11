# 2_6_heartbeat.py
import numpy as np
from darts.datasets import HeartRateDataset
import matplotlib.pyplot as plt
import nltk
from sklearn import preprocessing, model_selection
import keras


# 퀴즈
# HeartRateDataset 데이터에 대해 동작하는 RNN 모델을 구축하세요
# 70%로 학습하고, 30%에 대해 결과를 그래프로 보여주세요
def make_xy(seq_length):
    heart_rate = HeartRateDataset().load().pd_dataframe()
    # print(heart_rate)

    scaler = preprocessing.StandardScaler()
    values = scaler.fit_transform(heart_rate.values)

    data = nltk.ngrams(values, seq_length + 1)
    data = list(data)
    # print(data[0])

    # for item in data:
    #     print(np.array(item))

    # 퀴즈
    # y를 만드세요
    x = np.array([item[:-1] for item in data])
    y = np.array([item[-1] for item in data])
    print(x.shape, y.shape)     # (1797, 3, 1) (1797, 1)

    return x, y, scaler


x, y, scaler = make_xy(seq_length=3)

data = model_selection.train_test_split(x, y, train_size=0.7, shuffle=False)
x_train, x_test, y_train, y_test = data

model = keras.Sequential([
    keras.layers.SimpleRNN(16, return_sequences=False),
    keras.layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.mse,
              metrics='mae')

model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2)

p = model.predict(x_test, verbose=0)
print(p.shape)

# plt.plot(y_test, 'r')
# plt.plot(p, 'g')

yy = scaler.inverse_transform(y_test)
pp = scaler.inverse_transform(p)

plt.plot(yy, 'r')
plt.plot(pp, 'g')

# heart_rate.plot()
plt.show()

