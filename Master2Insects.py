import MatchFolderInsects as mt
from scipy.io import loadmat
import pandas as pd
import statistics as stats
import scipy.stats as stats1
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, kurtosis, entropy
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, losses, optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.stats import skew

# set the training dictionary
mt.getSelectedTrainingImages()

# then generate the datasets from the created csv
# set the model hyper parameters
BATCH_SIZE = 64
# set the image size to fit the resnet model for lower overfitting
IMG_SIZE = (224, 224)

# create training and validation image generator object and rescale the images,
# also splitting the data into training and validation sets
train_dataset_gen = ImageDataGenerator(rescale=1. / 255.,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.2
                                       )

train_df = pd.read_csv("csv/insects_train_data.csv")

# generate the datasets(training and validation)
training_dataset = train_dataset_gen.flow_from_dataframe(dataframe=train_df,
                                                         directory="./Data6/all_images/",
                                                         shuffle=True,
                                                         x_col="file",
                                                         y_col="label",
                                                         batch_size=BATCH_SIZE,
                                                         target_size=IMG_SIZE,
                                                         validate_filenames=False,
                                                         class_mode="categorical",
                                                         subset="training")

# generate the datasets(validation)
validation_dataset = train_dataset_gen.flow_from_dataframe(dataframe=train_df,
                                                           directory="./Data6/all_images/",
                                                           shuffle=True,
                                                           x_col="file",
                                                           y_col="label",
                                                           batch_size=BATCH_SIZE,
                                                           target_size=IMG_SIZE,
                                                           class_mode="categorical",
                                                           validate_filenames=False,
                                                           subset="validation")

print("training items:", training_dataset.samples)

print("validation items:", validation_dataset.samples)
