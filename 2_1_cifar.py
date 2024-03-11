# 2_1_cifar.py
import keras
import numpy as np


# 퀴즈
# cifar10 데이터에 대해 동작하는 모델을 구축하세요
cifar = keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar

print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)  # (50000, 1) (10000, 1)

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

print(np.min(y_train), np.max(y_train))     # 0 9
print(sorted(set(y_train)))                 # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(np.min(x_train), np.max(x_train))     # 0 255

x_train = x_train / 255
x_test = x_test / 255

# 1번
# model = keras.Sequential([
#     keras.layers.Input(shape=x_train.shape[1:]),
#     keras.layers.Conv2D(16, 3, 1, 'same', activation='relu'),
#     keras.layers.MaxPool2D(2, 2),
#
#     keras.layers.Flatten(),
#     keras.layers.Dense(10, activation='softmax'),
# ])

# 2번
# inputs = keras.layers.Input(shape=x_train.shape[1:])
# output = keras.layers.Conv2D(16, 3, 1, 'same', activation='relu')(inputs)
# output = keras.layers.MaxPool2D(2, 2)(output)
#
# output = keras.layers.Flatten()(output)
# output = keras.layers.Dense(10, activation='softmax')(output)
#
# model = keras.Model(inputs, output)

# 3번 - resnet 흉내
inputs = keras.layers.Input(shape=x_train.shape[1:])

output = keras.layers.Conv2D(8, 3, 1, 'same', activation='relu')(inputs)
output = keras.layers.Conv2D(32, 3, 1, 'same', activation='relu')(output)
shortcut = output

output = keras.layers.Conv2D(8, 1, 1, 'same', activation='relu')(output)
output = keras.layers.Conv2D(8, 3, 1, 'same', activation='relu')(output)
output = keras.layers.Conv2D(32, 1, 1, 'same', activation='relu')(output)

output = keras.layers.add([output, shortcut])
output = keras.layers.MaxPool2D(2, 2)(output)

output = keras.layers.GlobalAvgPool2D()(output)
# output = keras.layers.Flatten()(output)

output = keras.layers.Dense(10, activation='softmax')(output)

model = keras.Model(inputs, output)
model.summary()

model.compile(optimizer=keras.optimizers.RMSprop(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=1, verbose=2,
          batch_size=100, validation_data=(x_test, y_test))

p = model.predict(x_test, verbose=0)
p_arg = np.argmax(p, axis=1)
print(p.shape, p_arg.shape, y_test.shape)




