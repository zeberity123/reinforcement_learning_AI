# 5_8_tensorflow_hub.py
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
from keras.layers import BatchNormalization
import joblib
from tensorflow.keras.models import load_model


def data_generator():
    image_paths = [
        "UBC/train_thumbnails/1020_thumbnail.png",
        "UBC/train_thumbnails/1080_thumbnail.png",
        "UBC/train_thumbnails/10077_thumbnail.png",
        "UBC/train_thumbnails/10143_thumbnail.png",
        "UBC/train_thumbnails/10246_thumbnail.png",
        "UBC/train_thumbnails/10252_thumbnail.png",
        "UBC/train_thumbnails/10469_thumbnail.png",
        "UBC/train_thumbnails/10548_thumbnail.png",
        "UBC/train_thumbnails/10642_thumbnail.png",
        "UBC/train_thumbnails/10800_thumbnail.png",
    ]

    return np.array([np.array(Image.open(path).resize((256, 256))) for path in image_paths])


# Define a function to download the ResNet50 model from TensorFlow Hub
def download_resnet_model():
    resnet_url = "https://www.kaggle.com/models/tensorflow/resnet-50/frameworks/TensorFlow2/variations/classification/versions/1"  # ResNet50
    resnet_model = hub.KerasLayer(resnet_url, input_shape=(224, 224, 3), trainable=False,
                                  arguments=dict(weights='imagenet', include_top=False))
    resnet_model.trainable = False
    return resnet_model


def create_optimized_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


df = pd.read_csv('UBC/train.csv')

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(df['label'])
train_gen = data_generator()
print(train_gen.shape)
print(train_labels.shape)

# Create a new model with ResNet50 as the base
input_shape = (256, 256, 3)
num_classes = len(label_encoder.classes_)
model = create_optimized_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, train_labels, epochs=2)

