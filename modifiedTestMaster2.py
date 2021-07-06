import MatcherFolder as mt
import statistics as stats
from collections import defaultdict
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, losses, optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# set the training dictionary
mt.getSelectedTrainingImages()

# set the validation dictionary
mt.getSelectedValidationImages()

# set the model hyper parameters
BATCH_SIZE = 32
# set the image size to fit the resnet model for lower overfitting
IMG_SIZE = (224, 224)

# create training and validation image generator object and rescale the images
train_dataset_gen = ImageDataGenerator(rescale=1. / 255.)
validation_dataset_gen = ImageDataGenerator(rescale=1. / 255.)

train_df = pd.read_csv("csv/train_data.csv")
validation_df = pd.read_csv("csv/validation_data.csv")

# generate the datasets(training and validation)
training_dataset = train_dataset_gen.flow_from_dataframe(dataframe=train_df,
                                                         directory="./Data/train/",
                                                         x_col="file",
                                                         y_col="label",
                                                         shuffle=True,
                                                         batch_size=BATCH_SIZE,
                                                         target_size=IMG_SIZE,
                                                         class_mode="sparse",
                                                         subset="training")

validation_dataset = validation_dataset_gen.flow_from_dataframe(dataframe=validation_df,
                                                                directory="./Data/validate/",
                                                                x_col="file",
                                                                y_col="label",
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                target_size=IMG_SIZE,
                                                                class_mode="sparse",
                                                                subset=None)

class_names = training_dataset.class_indices
# reverse the dictionary
new_dict = {}
for key, value in class_names.items():
    new_dict[value] = key

for _ in range(len(training_dataset.filenames)):
    image, label = training_dataset.next()

    print(new_dict[label[0]])
    # display the image from the iterator
    # plt.imshow(image[0])
    # label_name=new_dict[label[0]] # note you are only showing the first image of the batch
    # plt.title(label_name)
    # plt.show()

# get the pretrained model
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
    # print(lyr, "Layer prob:", val)
    if v > selected_layers_mean:
        # store the probabilities of the upper half selected convolved layers
        final_selected_layers.append(s_lyr)

# print(second_half_layers)
# print(str(convolved_layers[c_layer]) + " probability: " + str(layer_pos_prob))
# print("*******END LAYER: " + str(convolved_layers[c_layer]) + " **************** ")

# use the selected layers
for sb_layer in model.layers[:-2]:
    #sb_layer.trainable = False
    index = getLayerIndex(model, sb_layer.name)
    for b in final_selected_layers:
        if b == index:
            sb_layer.trainable = True
            print(str(sb_layer.name) + " and index is"+str(b))


# try the transfer learning model
to_res = (224, 224)

t_model = Sequential()
t_model.add(model)
t_model.add(layers.Flatten())
t_model.add(layers.BatchNormalization())
t_model.add(layers.Dense(256, activation='relu'))
t_model.add(layers.Dropout(0.5))
t_model.add(layers.BatchNormalization())
t_model.add(layers.Dense(128, activation='relu'))
t_model.add(layers.Dropout(0.5))
t_model.add(layers.BatchNormalization())
t_model.add(layers.Dense(64, activation='relu'))
t_model.add(layers.Dropout(0.5))
t_model.add(layers.BatchNormalization())
t_model.add(layers.Dense(2, activation='softmax'))

t_model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=optimizers.Adam(lr=3e-4),
              metrics=['accuracy'])
history = t_model.fit(training_dataset, batch_size=32, epochs=10, verbose=1)

# create a method for prediction
# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img


print("prediction with the test data")
t_model.evaluate(validation_dataset, batch_size=32, verbose=2)

#do some prediction
probability_model = Sequential([t_model, layers.Softmax()])

print("see the results for an image")
# load the image
images_array=['bike_242.bmp','carsgraz_253.bmp','bike_247.bmp','bike_244.bmp'];

for i in range(len(images_array)):

    image1 = load_image(images_array[i])
    predictions = probability_model.predict(image1)

    print("The prediction class for:"+images_array[i]+" is:")
    predicted_class_indices = np.argmax(predictions, axis=1)
    training_labels = training_dataset.class_indices
    labels = dict((v,k) for k,v in training_labels.items())
    predictions_y = [labels[k] for k in predicted_class_indices]

    print(predictions_y)