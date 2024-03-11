# 2_5_mnist.py
import keras


# 퀴즈
# mnist 데이터에 대해 동작하는 RNN 모델을 만드세요
mnist = keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist

print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)  # (60000,) (10000,)

x_train = x_train / 255
x_test = x_test / 255

# 1번 - basic
# x_train = x_train.reshape(-1, 784)
# x_test = x_test.reshape(-1, 784)
#
# model = keras.Sequential([
#     keras.layers.Dense(10, activation='softmax')
# ])

# 2번 - cnn
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
#
# model = keras.Sequential([
#     keras.layers.Conv2D(6, 3, 1, 'same', activation='relu'),
#     keras.layers.MaxPool2D(2, 2),
#     keras.layers.Conv2D(6, 3, 1, 'same', activation='relu'),
#     keras.layers.MaxPool2D(2, 2),
#     keras.layers.Flatten(),
#     keras.layers.Dense(10, activation='softmax'),
# ])

# 3번 - rnn
x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

model = keras.Sequential([
    keras.layers.SimpleRNN(16),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=10, batch_size=100,
          verbose=2, validation_data=(x_test, y_test))

