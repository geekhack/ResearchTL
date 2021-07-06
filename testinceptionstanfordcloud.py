import pandas as pd
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import os.path
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3 as inception
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2 as l2
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.layers import Dropout,GlobalAveragePooling2D
from tensorflow.python.keras.models import Sequential

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

BATCH_SIZE = 16
# set the image size to fit the resnet model for lower overfitting
IMG_SIZE = (224, 224)

train_dataset_gen = ImageDataGenerator(rescale=1. / 255.
                                       )
train_df = pd.read_csv("skew_stanford_training_data.csv")

training_dataset = train_dataset_gen.flow_from_dataframe(dataframe=train_df,
                                                         directory="new_stanford_cars/",
                                                         shuffle=True,
                                                         x_col="file",
                                                         y_col="label",
                                                         batch_size=BATCH_SIZE,
                                                         target_size=IMG_SIZE,
                                                         validate_filenames=False,
                                                         class_mode="categorical",
                                                         subset="training")

# validation_dataset = train_dataset_gen.flow_from_dataframe(dataframe=train_df,
#                                                          directory="../flowers/",
#                                                          shuffle=True,
#                                                          x_col="file",
#                                                          y_col="label",
#                                                          batch_size=8,
#                                                          target_size=IMG_SIZE,
#                                                          validate_filenames=False,
#                                                          class_mode="categorical",
#                                                          subset="validation")
model_name = inception
input_t = keras.Input(shape=(224, 224, 3))
model = model_name(include_top=False,
                   weights="imagenet",
                   input_tensor=input_t)

for layer in model.layers[:-2]:
    layer.trainable = False

# try the transfer learning model
to_res = (224, 224)

t_model = Sequential()
t_model.add(model)
t_model.add(GlobalAveragePooling2D())
t_model.add(Dropout(0.5))
# t_model.add(layers.BatchNormalization())
# t_model.add(layers.Dense(256, activation='relu'))
# t_model.add(layers.Dropout(0.5))
# t_model.add(layers.BatchNormalization())
# t_model.add(layers.Dense(128, activation='relu'))
# t_model.add(layers.Dropout(0.5))
# t_model.add(layers.BatchNormalization())
# t_model.add(layers.Dense(64, activation='relu'))
# t_model.add(layers.Dropout(0.5))
# t_model.add(layers.BatchNormalization())
t_model.add(layers.Dense(196, activation='softmax'))

t_model.compile(loss=losses.CategoricalCrossentropy(from_logits=True),
                optimizer=optimizers.SGD(lr=1e-4,
                                       momentum=0.9),
                metrics=['accuracy'])
#improves history = t_model.fit(training_dataset, batch_size=32, epochs=100, verbose=1, callbacks=[callback])

history = t_model.fit(training_dataset, batch_size=32, epochs=30, verbose=1)
#
# history2 = t_model.fit(validation_dataset, batch_size=32, epochs=20, verbose=1)
#
# def load_image(filename):
#     # load the image
#     img = load_img(filename, target_size=(224, 224))
#     # convert to array
#     img = img_to_array(img)
#     # reshape into a single sample with 3 channels
#     img = img.reshape(1, 224, 224, 3)
#     # center pixel data
#     img = img.astype('float32')
#     img = img - [123.68, 116.779, 103.939]
#     return img
#
#
#
# # do some prediction
# probability_model = Sequential([t_model, layers.Softmax()])
#
# print("see the results for an image")
# # load the image
# images_array = ['Laysan_Albatross_0068_726.jpg', 'Bobolink_0047_9204.jpg'];
#
# for i in range(len(images_array)):
#     image1 = load_image(images_array[i])
#     predictions = probability_model.predict(image1)
#
#     print("The prediction class for:" + images_array[i] + " is:")
#     predicted_class_indices = np.argmax(predictions, axis=1)
#     training_labels = training_dataset.class_indices
#     labels = dict((v, k) for k, v in training_labels.items())
#     predictions_y = [labels[k] for k in predicted_class_indices]
#
#     print(predictions_y)
