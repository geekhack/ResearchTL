import MatcherFolder as mt
import tensorflow as tf
import pandas as pd
import DataSetCreator
import numpy as np
from tensorflow.keras.preprocessing import image as Img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# set the training dictionary
mt.getSelectedTrainingImages()

datagen = ImageDataGenerator()

train_df_2 = pd.read_csv("data.csv")
train_generator = datagen.flow_from_dataframe(dataframe=train_df_2,
                                              directory="Data/train",
                                              x_col='image',
                                              y_col="class")

print(train_generator.class_indices)