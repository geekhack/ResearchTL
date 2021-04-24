import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50

input_t = keras.Input(shape=(224, 224, 3))
res_model = ResNet50(include_top=False,
                     weights="imagenet",
                     input_tensor=input_t)

print("The weights shape of layer 2 is:")
print(res_model.layers[2].get_weights()[0].shape)

for i in range(len(res_model.layers[2].get_weights()[0])):
    # get the first convolutional layer
    if i == 1:
        # access the weights in the first layer
        print("*********************LAYER: " + str(i) + "****************************")
        for x in range(len(res_model.layers[2].get_weights()[0][i])):
            # create an array of weights in features of the image that are non-negative
            x1 = np.array(res_model.layers[2].get_weights()[0][i][x]) > 0
            # in each feature loop through it and return the weights for each channel
            for a in range(len(x1)):
                print("all the trues in channel" + str(a + 1) + " are for feature " + str(x + 1))
                print ("the count of the +ve values in each channel")
                print(sum(x1[a]))
        print("********************* END OF LAYER: " + str(i) + "****************************")

    else:
        print("order:")

print("************************for the next layer*******************************************")
