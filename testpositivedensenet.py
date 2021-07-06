#import MatcherFolder as mt
import statistics as stats
from collections import defaultdict
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.inception_v3 import InceptionV3 as inception
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16
from tensorflow.keras.applications.densenet import DenseNet169 as densenet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.layers import Dropout,GlobalAveragePooling2D,Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

# set the training dictionary
#mt.getSelectedTrainingImages()

# set the validation dictionary
#mt.getSelectedValidationImages()

# set the model hyper parameters
BATCH_SIZE = 16
# set the image size to fit the resnet model for lower overfitting
IMG_SIZE = (224, 224)

# create training and validation image generator object and rescale the images

train_dataset_gen = ImageDataGenerator(rescale=1. / 255,
                                       validation_split=0.2
                                       )
#validation_dataset_gen = ImageDataGenerator(rescale=1. / 255.)

#train_df = pd.read_csv("csv/train_data.csv")
#validation_df = pd.read_csv("csv/validation_data.csv")

# generate the datasets(training and validation)
#train_df = pd.read_csv("csv/skew_flowers_data.csv")
train_df = pd.read_csv("csv/merged_flowers.csv")

training_dataset = train_dataset_gen.flow_from_dataframe(dataframe=train_df,
                                                         directory="Data3/train/",
                                                         shuffle=True,
                                                         x_col="file",
                                                         y_col="label",
                                                         batch_size=BATCH_SIZE,
                                                         target_size=IMG_SIZE,
                                                         validate_filenames=False,
                                                         class_mode="categorical",
                                                         subset="training")
validation_dataset = train_dataset_gen.flow_from_dataframe(dataframe=train_df,
                                                         directory="Data3/train/",
                                                         shuffle=True,
                                                         x_col="file",
                                                         y_col="label",
                                                         batch_size=BATCH_SIZE,
                                                         target_size=IMG_SIZE,
                                                         validate_filenames=False,
                                                         class_mode="categorical",
                                                         subset="validation")

model_name = densenet
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

# create a dictionary for storing the layers and their +ve values probabilities
layer_probs_dict = defaultdict(list)

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
        # print(key, "corresponds to:",val," positive values")
        positives_array.append(val)
        ###used with softmax values
        # print(key, "corresponds to:", float(val))
        # positives_array.append(float(val))
        ###end of usage with softmax
        # create an array of positives and then use the softmax to get their distribution
        # probability

    # get the softmax values
    # p = tf.nn.softmax(positives_array)
    # print(p)
    list_layer_probs = []
    list_layer_probs.append((str(convolved_layers[c_layer]), layer_pos_prob))
    layer_probs_dict.update(list_layer_probs)

    # loop through the layer_probs_dictionary
    for lyr, val in layer_probs_dict.items():
        print(lyr, "Layer prob:", val)

# get the median number of layers to ensure the first layers deal with the feature extraction

median_layer = stats.median(convolved_layers)

# create dictionary for storing selected median layers
second_layer_probs_dict = defaultdict(list)
# store the new list of layers to be matched with the mean probability
second_half_layers = []

# store all probabilities for the selected upper half layers
second_half_probs = []
# loop through the layers and print those layers above the median
for lyr, val in layer_probs_dict.items():
    # print(lyr, "Layer prob:", val)
    if int(lyr) > median_layer:
        # store the probabilities of the upper half selected convolved layers
        print(lyr, "Layer prob:", val)
        # update the sum of selected layers probabilities
        second_half_probs.append(val)
        second_half_layers.append((lyr, val))
        second_layer_probs_dict.update(second_half_layers)

# get the mean of the layers
selected_layers_mean = stats.mean(second_half_probs)
final_selected_layers = []
# now get the final layers list whose value exceed the mean
for s_lyr, v in second_layer_probs_dict.items():
    # get the probabilities that are lower than the mean probability
    if v < selected_layers_mean:
        # store the probabilities of the upper half selected convolved layers
        final_selected_layers.append(s_lyr)

# print(second_half_layers)
# print(str(convolved_layers[c_layer]) + " probability: " + str(layer_pos_prob))
# print("*******END LAYER: " + str(convolved_layers[c_layer]) + " **************** ")

# use the selected layers
for sb_layer in model.layers[:-2]:
    # sb_layer.trainable = False
    index = getLayerIndex(model, sb_layer.name)
    for b in final_selected_layers:
        if b == index:
            sb_layer.trainable = True
            print(str(sb_layer.name) + " and index is" + str(b))

# try the transfer learning model
to_res = (224, 224)

t_model = Sequential()
t_model.add(model)
t_model.add(GlobalAveragePooling2D())
t_model.add(Dropout(0.5))
# t_model.add(layers.BatchNormalization())
# t_model.add(layers.Dense(256, activation='relu'))
# t_model.add(layers.Dropout(0.5))
# t_model.add(layers.BatchNormalization())
# t_model.add(layers.Dense(128, activation='relu'))
# t_model.add(layers.Dropout(0.5))
# t_model.add(layers.BatchNormalization())
# t_model.add(layers.Dense(64, activation='relu'))
# t_model.add(layers.Dropout(0.5))
# t_model.add(layers.BatchNormalization())
t_model.add(layers.Dense(5, activation='softmax'))

t_model.compile(loss=losses.CategoricalCrossentropy(from_logits=True),
                optimizer=optimizers.SGD(lr=1e-4,
                                       momentum=0.9),
                metrics=['accuracy'])
history = t_model.fit(training_dataset, batch_size=16, shuffle=True,validation_data=validation_dataset, epochs=10, verbose=1)
# t_model.evaluate(validation_dataset,batch_size=32)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
