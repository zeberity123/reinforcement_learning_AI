# 2_4_flower17.py
import keras

gen_train = keras.preprocessing.image.ImageDataGenerator(rescale=1/255, rotation_range=30)
flow_train = gen_train.flow_from_directory('flower17/train',
                                           target_size=(224, 224),
                                           class_mode='sparse',
                                           batch_size=16)

gen_test = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
flow_test = gen_test.flow_from_directory('flower17/test',
                                         target_size=(224, 224),
                                         class_mode='sparse',
                                         batch_size=16)

# model = keras.Sequential([
#     keras.layers.Input(shape=(224, 224, 3)),
#     keras.layers.Conv2D(8, 3, 1, 'same', activation='relu'),
#     keras.layers.MaxPool2D(2, 2),
#
#     keras.layers.Conv2D(16, 3, 1, 'same', activation='relu'),
#     keras.layers.MaxPool2D(2, 2),
#
#     keras.layers.Conv2D(32, 3, 1, 'same', activation='relu'),
#     keras.layers.MaxPool2D(2, 2),
#
#     keras.layers.Conv2D(64, 3, 1, 'same', activation='relu'),
#     keras.layers.MaxPool2D(2, 2),
#
#     keras.layers.Conv2D(64, 3, 1, 'same', activation='relu'),
#     keras.layers.MaxPool2D(2, 2),
#
#     keras.layers.Flatten(),
#
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(3, activation='softmax'),
# ])

conv_base = keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3))
conv_base.trainable = False

model = keras.Sequential([
    conv_base,
    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3, activation='softmax'),
])
model.summary()

model.compile(optimizer=keras.optimizers.RMSprop(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit(flow_train, epochs=10, verbose=2, validation_data=flow_test)
