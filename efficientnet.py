import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import EfficientNetB1 as efficient
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16

model_name=efficient
input_t = keras.Input(shape=(224, 224, 3))
model = model_name(include_top=False,
                     weights="imagenet",
                     input_tensor=input_t)
#print(res_model.summary())
# get the layer index
def getLayerIndex(model, layer_name):
    for pos, layer in enumerate(model.layers):
        if layer.name == layer_name:
            return pos


for layer in model.layers[:-2]:
    x = np.array(layer.get_weights()).ndim
    array = np.array(layer.get_weights())
    if (model_name != resnet50) or (model_name != vgg16):
        if (len(array) > 0) and (x > 2):
            index = getLayerIndex(model, layer.name)
            print(str(len(array)) + "for:" + layer.name + "at index:" + str(index))
    if (model_name == resnet50) or (model_name == vgg16):
        if len(array) > 0 and (x != 2):
            index = getLayerIndex(model, layer.name)
            print(str(len(array)) + "for:" + layer.name + "at index:" + str(index))