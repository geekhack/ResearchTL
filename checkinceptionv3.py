import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3 as inceptionv3
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16

model_name=inceptionv3
input_t = keras.Input(shape=(224, 224, 3))
model = model_name(include_top=False,
                     weights="imagenet",
                     input_tensor=input_t)
print(model.summary())
for layer in model.layers[:-2]:
    x = np.array(layer.get_weights()).ndim
    array = np.array(layer.get_weights())
    #evaluate that the length of array is greater than 0(empty arrays that correspond to activation
    #layers and that it is ndim is not 2 since that represents batch normalization layers
    if (len(array) > 0) and (x != 2):
        print(str(len(array)) + "for:" + layer.name)