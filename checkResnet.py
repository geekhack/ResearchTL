import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16

model_name=resnet50
input_t = keras.Input(shape=(224, 224, 3))
model = model_name(include_top=False,
                     weights="imagenet",
                     input_tensor=input_t)
#print(res_model.summary())
for layer in model.layers[:-2]:
    #print(layer.name)
    x = np.array(layer.get_weights()).ndim
    array = np.array(layer.get_weights())
    if ((model_name == resnet50) or (model_name == vgg16)):

        if (len(array) > 0) and (x >= 3):
            print(str(len(array)) + "for:" + layer.name)

    elif ((model_name == resnet50) or (model_name == vgg16)):
        if (len(array) > 0) and (x !=2):
            print(str(len(array)) + "for:" + layer.name)

    #x = np.array(layer.get_weights()).shape
    # if x == 1:
    #     #check if value contained is 0 or not
    #     m=np.array([0,])
    #     if(array == m.shape):
    #         print(array)