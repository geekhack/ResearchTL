# import MatcherFolder as mt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16
import matplotlib.pyplot as plt
from tensorflow.keras import backend as k
from collections import defaultdict

model_name = resnet50
input_t = keras.Input(shape=(224, 224, 3))
model = model_name(include_top=False,
                   weights="imagenet",
                   input_tensor=input_t)


# get the layer index
def getLayerIndex(model_i, layer_name):
    for pos, layer_g in enumerate(model_i.layers):
        if layer_g.name == layer_name:
            return pos


# get the convolved layers into an array for looping
convolved_layers = []

for layer in model.layers[:-2]:

    t = np.array(layer.get_weights()).ndim
    array = np.array(layer.get_weights())
    if (model_name != resnet50) or (model_name != vgg16):
        if (len(array) > 0) and (t > 2):
            index = getLayerIndex(model, layer.name)
            # append the convolved layer
            convolved_layers.append(index)
            # print(str(len(array)) + "for:" + layer.name + "at index:" + str(index))
    if (model_name == resnet50) or (model_name == vgg16):
        if len(array) > 0 and (t != 2):
            index = getLayerIndex(model, layer.name)
            # append the convolved layer
            convolved_layers.append(index)
            # print(str(len(array)) + "for:" + layer.name + "at index:" + str(index))

# get the total number of weights in the array
total_layer_weights = 0
sum_positives = 0
sum_negatives = 0

# create a dictionary with layer index and positive values identified
layer_positives_dict = defaultdict(list)

# list the convolved layers
for c_layer in range(len(convolved_layers)):
    # then print for each layer
    ####print("*******START LAYER: " + str(convolved_layers[c_layer]) + " **************** ")
    ### print("Batches:" + str(len(model.layers[convolved_layers[c_layer]].get_weights()[0])));
    ####print("Layer output size: " + str(model.layers[convolved_layers[c_layer]].get_weights()[0].shape))

    # create array for array matrices,feature values and convolved values for each layer
    layer_array_matrices = []
    layer_feature_sums = []
    layer_convolved_values_sum = []

    # get the arrays matrices(batches) pixel
    for i in range(len(model.layers[convolved_layers[c_layer]].get_weights()[0])):
        # get the values for each feature
        ######print("channels(image):"+str(len(model.layers[convolved_layers[c_layer]].get_weights()[0][i])))
        for x in range(len(model.layers[convolved_layers[c_layer]].get_weights()[0][i])):
            ####print("Convolved image array :"+str(len(np.array(model.layers[convolved_layers[c_layer]].get_weights()[0][i][x]))))

            # return all the values in the convolutional feature
            total_array = np.array(model.layers[convolved_layers[c_layer]].get_weights()[0][i][x])
            # return all the negative values in the convolutional feature
            x2 = np.array(model.layers[convolved_layers[c_layer]].get_weights()[0][i][x]) < 0
            for c in range(len(x2)):
                ###print("all the trues in depth: " + str(c + 1) + " are for feature: " + str(x + 1))
                sum_negatives += sum(x2[c])
                # print(sum(x2[c]))
            # return all the values
            for y in range(len(total_array)):
                total_layer_weights += len(total_array[y])

            x1 = np.array(model.layers[convolved_layers[c_layer]].get_weights()[0][i][x]) > 0
            # get the values for each channel
            for a in range(len(x1)):
                # print("Pixel(matrix position): " + str(a + 1) + " Feature Map item: " + str(x + 1)+" Array: "+str(i+1)+" of "+str(len(model.layers[convolved_layers[c_layer]].get_weights()[0])))
                # print("+ve image convolved values at position:" + str(a))
                # get the positive values ready to be passed through activation function for feature map
                value_pix = sum(x1[a])
                # get the positive values
                sum_positives += value_pix

                ######print(value_pix)
    list_sums = []
    list_sums.append((str(convolved_layers[c_layer]), sum_positives))
    layer_positives_dict.update(list_sums)

    # print the probabilities for each layer
    layer_pos_prob = sum_positives / total_layer_weights

    # create +ves values array
    positives_array = []
    for key, val in layer_positives_dict.items():
        print(key, "corresponds to:", val)
        positives_array.append(val)
        # create an array of positives and then use the softmax to get their distribution
        # probability

    #get the softmax values
    p = tf.nn.softmax(positives_array)
    print(p)

    ####print(str(convolved_layers[c_layer]) + " probability: " + str(layer_pos_prob))
    ##print("*******END LAYER: " + str(convolved_layers[c_layer]) + " **************** ")
