import MatcherFolder as mt
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# set the training dictionary
mt.getSelectedTrainingImages()

# set the validation dictionary
mt.getSelectedValidationImages()

# set the model hyper parameters
BATCH_SIZE = 32
# set the image size to fit the resnet model for lower overfitting
IMG_SIZE = (28, 28)

# create training and validation image generator object and rescale the images
train_dataset_gen = ImageDataGenerator(rescale=1. / 255.)
validation_dataset_gen = ImageDataGenerator(rescale=1. / 255.)

train_df = pd.read_csv("csv/train_data.csv")
validation_df = pd.read_csv("csv/validation_data.csv")

train_file_paths = train_df['file'].values;
train_file_labels = train_df['label'].values;

ds_train = tf.data.Dataset.from_tensor_slices((train_file_paths, train_file_labels))

# read an image
directory = "./Data/train/";


def read_image(image_file, label):
    image = tf.io.read_file(directory + image_file)
    image = tf.image.decode_bmp(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
   # image = tf.image.resize(image, [28, 28])
    image = tf.reshape(image, [28, 28, 3])
    return image, label


# for image, label in ds_train.take(1):
#     x = read_image(image, label);
#     print(x[1])


def augment(image, label):
    # data augmentation here
    return image, label


ds_train = ds_train.map(read_image)

inputs = keras.Input(shape=(28, 28, 3))
x = layers.Dense(512, activation='relu', name='first_layer')(inputs)
x = layers.Dense(256, activation='relu', name='second_layer')(x)
x = layers.Flatten()(x)
output = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=output)
print(model.summary())

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(lr=3e-4),
              metrics=['accuracy'])

model.fit(ds_train, epochs=10, verbose=2)
