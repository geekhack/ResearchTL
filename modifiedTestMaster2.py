# import MatcherFolder as mt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16
import matplotlib.pyplot as plt
from tensorflow.keras import backend as k

model_name = resnet50
input_t = keras.Input(shape=(224, 224, 3))
model = model_name(include_top=False,
                   weights="imagenet",
                   input_tensor=input_t)


# get the layer index
def getLayerIndex(model_i, layer_name):
    for pos, layer in enumerate(model_i.layers):
        if layer.name == layer_name:
            return pos


# get the convolved layers into an array for looping
convolved_layers = []

for layer in model.layers[:-2]:

    t = np.array(layer.get_weights()).ndim
    array = np.array(layer.get_weights())
    if (model_name != resnet50) or (model_name != vgg16):
        if (len(array) > 0) and (t > 2):
            index = getLayerIndex(model, layer.name)
            #append the convolved layer
            convolved_layers.append(index)
            #print(str(len(array)) + "for:" + layer.name + "at index:" + str(index))
    if (model_name == resnet50) or (model_name == vgg16):
        if len(array) > 0 and (t != 2):
            index = getLayerIndex(model, layer.name)
            # append the convolved layer
            convolved_layers.append(index)
            #print(str(len(array)) + "for:" + layer.name + "at index:" + str(index))


#list the convolved layers
for c_layer in range(len(convolved_layers)):
    #then print for each layer
    print("*******START LAYER: "+ str(convolved_layers[c_layer])+ " **************** ")
    print("Batches:" + str(len(model.layers[convolved_layers[c_layer]].get_weights()[0])));
    print("Layer output size: " + str(model.layers[convolved_layers[c_layer]].get_weights()[0].shape))
    #get the arrays matrices(batches) pixel
    for i in range(len(model.layers[convolved_layers[c_layer]].get_weights()[0])):
        # get the values for each feature
        print("channels(image):"+str(len(model.layers[convolved_layers[c_layer]].get_weights()[0][i])))
        for x in range(len(model.layers[convolved_layers[c_layer]].get_weights()[0][i])):
            print("image pixels :"+str(len(np.array(model.layers[convolved_layers[c_layer]].get_weights()[0][i][x]))))
            x1 = np.array(model.layers[convolved_layers[c_layer]].get_weights()[0][i][x]) > 0
            # get the values for each channel
            for a in range(len(x1)):
                print("Pixel(matrix position): " + str(a + 1) + " Feature Map item: " + str(x + 1)+" Array: "+str(i+1)+" of "+str(len(model.layers[convolved_layers[c_layer]].get_weights()[0])))
                print("+ve image convolved values at position:" + str(a))
                print(sum(x1[a]))
    print("*******END LAYER: " + str(convolved_layers[c_layer]) + " **************** ")