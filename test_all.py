# import MatcherFolder as mt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, losses, optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
from collections import defaultdict
import matplotlib.pyplot as plt
from tensorflow.keras import backend as k


input_t = keras.Input(shape=(224, 224, 3))
res_model = ResNet50(include_top=False,
                     weights="imagenet",
                     input_tensor=input_t)

# get the total number of weights in the array
total_layer_weights = 0
sum_positives = 0
sum_negatives = 0

#create a dictionary with layer index and positive values identified
layer_positives_dict = defaultdict(list)

for i in range(len(res_model.layers[2].get_weights()[0])):
    # if i == 1:

    for x in range(len(res_model.layers[2].get_weights()[0][i])):
        # return all the values in the convolutional feature
        total_array = np.array(res_model.layers[2].get_weights()[0][i][x])
        # return all the negative values in the convolutional feature
        x2 = np.array(res_model.layers[2].get_weights()[0][i][x]) < 0
        for c in range(len(x2)):
            print("all the trues in depth: " + str(c + 1) + " are for feature: " + str(x + 1))
            sum_negatives += sum(x2[c])
            print(sum(x2[c]))

        for y in range(len(total_array)):
            total_layer_weights += len(total_array[y])

        # return all the positive values in the convolutional feature
        x1 = np.array(res_model.layers[2].get_weights()[0][i][x]) > 0

        for a in range(len(x1)):
            print("all the trues in depth: " + str(a + 1) + " are for feature: " + str(x + 1))
            sum_positives += sum(x1[a])
            print(sum(x1[a]))
    list_sums = []
    list_sums.append ((2,sum_positives))
    layer_positives_dict.update(list_sums)
# else:
#    print("order:")


for key,val in layer_positives_dict.items():
    print(key,"corresponds to:",val)

print("total weights for layer:" + str(total_layer_weights))
print("total positives:" + str(sum_positives))
print("total negatives:" + str(sum_negatives))
layer_pos_prob = sum_positives / total_layer_weights
print("probability: " + str(layer_pos_prob))