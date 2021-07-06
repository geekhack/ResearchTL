#import MatcherFolder as mt
import statistics as stats
from collections import defaultdict
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.inception_v3 import InceptionV3 as inception
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16
from tensorflow.keras.applications.densenet import DenseNet169 as densenet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.layers import Dropout,GlobalAveragePooling2D,Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

# set the training dictionary
#mt.getSelectedTrainingImages()

# set the validation dictionary
#mt.getSelectedValidationImages()

# set the model hyper parameters
BATCH_SIZE = 16
# set the image size to fit the resnet model for lower overfitting
IMG_SIZE = (224, 224)

# create training and validation image generator object and rescale the images

train_dataset_gen = ImageDataGenerator(rescale=1. / 255,
                                       validation_split=0.2
                                       )
#validation_dataset_gen = ImageDataGenerator(rescale=1. / 255.)

#train_df = pd.read_csv("csv/train_data.csv")
#validation_df = pd.read_csv("csv/validation_data.csv")

# generate the datasets(training and validation)
#train_df = pd.read_csv("csv/skew_flowers_data.csv")
train_df = pd.read_csv("csv/merged_flowers.csv")

training_dataset = train_dataset_gen.flow_from_dataframe(dataframe=train_df,
                                                         directory="Data3/train/",
                                                         shuffle=True,
                                                         x_col="file",
                                                         y_col="label",
                                                         batch_size=BATCH_SIZE,
                                                         target_size=IMG_SIZE,
                                                         validate_filenames=False,
                                                         class_mode="categorical",
                                                         subset="training")
validation_dataset = train_dataset_gen.flow_from_dataframe(dataframe=train_df,
                                                         directory="Data3/train/",
                                                         shuffle=True,
                                                         x_col="file",
                                                         y_col="label",
                                                         batch_size=BATCH_SIZE,
                                                         target_size=IMG_SIZE,
                                                         validate_filenames=False,
                                                         class_mode="categorical",
                                                         subset="validation")

model_name = densenet
input_t = keras.Input(shape=(224, 224, 3))
model = model_name(include_top=False,
                   weights="imagenet",
                   input_tensor=input_t)

for layer in model.layers:
    layer.trainable=False

# try the transfer learning model
to_res = (224, 224)

t_model = Sequential()
t_model.add(model)
t_model.add(Flatten())
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
t_model.add(layers.Dense(5, activation='softmax'))

t_model.compile(loss=losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'],
                optimizer=optimizers.SGD(lr=1e-4,
                                       momentum=0.9))
#improves history = t_model.fit(training_dataset, batch_size=32, epochs=100, verbose=1, callbacks=[callback])

history = t_model.fit(training_dataset, batch_size=16, shuffle=True,validation_data=validation_dataset, epochs=10, verbose=1)
# t_model.evaluate(validation_dataset,batch_size=32)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
