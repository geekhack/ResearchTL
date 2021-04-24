import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.nasnet import NASNetLarge as nasnet
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16

model_name=nasnet
input_t = keras.Input(shape=(224, 224, 3))
model = model_name(include_top=False,
                     weights="imagenet",
                     input_tensor=input_t)
#print(res_model.summary())
for layer in model.layers[:-2]:

    x = np.array(layer.get_weights()).ndim
    # array=np.array(layer.get_weights())
    # #if (len(array) > 0) and (x != 2): #for resnet50
    # if (len(array) > 0) and (x != 2):  # for resnet50
    #     print(str(len(array))+"for:"+layer.name)
    #x = np.array(layer.get_weights()).shape
    # if x == 1:
    #     #check if value contained is 0 or not
    #     m=np.array([0,])
    #     if(array == m.shape):
    #         print(array)