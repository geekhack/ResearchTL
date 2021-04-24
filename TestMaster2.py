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
import matplotlib.pyplot as plt
from tensorflow.keras import backend as k

# # set the training dictionary
# mt.getSelectedTrainingImages()
#
# # set the validation dictionary
# mt.getSelectedValidationImages()
#
# # set the model hyper parameters
# BATCH_SIZE = 32
# # set the image size to fit the resnet model for lower overfitting
# IMG_SIZE = (224, 224)
#
# # create training and validation image generator object and rescale the images
# train_dataset_gen = ImageDataGenerator(rescale=1. / 255.)
# validation_dataset_gen = ImageDataGenerator(rescale=1. / 255.)
#
# train_df = pd.read_csv("train_data.csv")
# validation_df = pd.read_csv("validation_data.csv")
#
# # data(training and validation) preprocess - show the first 3 images
# # print("*********training data****************")
# # print(train_df.head(3))
# #
# # print("*********validation data**************")
# # print(validation_df.head())
#
# # generate the datasets(training and validation)
# training_dataset = train_dataset_gen.flow_from_dataframe(dataframe=train_df,
#                                                          directory="./Data/train/",
#                                                          x_col="file",
#                                                          y_col="label",
#                                                          shuffle=True,
#                                                          batch_size=BATCH_SIZE,
#                                                          target_size=IMG_SIZE,
#                                                          class_mode="sparse",
#                                                          subset="training")
#
# validation_dataset = validation_dataset_gen.flow_from_dataframe(dataframe=validation_df,
#                                                                 directory="./Data/validate/",
#                                                                 x_col="file",
#                                                                 y_col="label",
#                                                                 shuffle=True,
#                                                                 batch_size=BATCH_SIZE,
#                                                                 target_size=IMG_SIZE,
#                                                                 class_mode="sparse",
#                                                                 subset=None)
#
# class_names = training_dataset.class_indices
#
# # reverse the dictionary
# new_dict = {}
# for key, value in class_names.items():
#     new_dict[value] = key
#
# for _ in range(len(training_dataset.filenames)):
#     image, label = training_dataset.next()
#     print(new_dict[label[0]])
# display the image from the iterator
# plt.imshow(image[0])
# label_name=new_dict[label[0]] # note you are only showing the first image of the batch
# plt.title(label_name)
# plt.show()

#
# for _ in range(len(training_dataset.filenames)):
#     file, label = training_dataset.next()
#     # check if the image shapes match those of resnet 50
#
#     # display the first image from the iterator
#     plt.imshow(file[0])
#     plt.show()


# show the classes for both training and validation sets
# print(list(training_dataset.class_indices.keys()))
# print(validation_dataset.class_indices)
#
# # show the images filenames
# print("*************training images in dataset****************")
# print(training_dataset.filenames)
# print("*************validation images in dataset**************")
# print(validation_dataset.filenames)
#
# # loop through the training images
#
# print("***************Training images***************")
# # get the labels
#
# t_labels = []
# for _ in range(len(training_dataset.filenames)):
#     image, label = training_dataset.next()
#     # check if the image shapes match those of resnet 50
#     t_labels.append(label)
#     # display the first image from the iterator
#     plt.imshow(image[0])
#     plt.title(training_dataset.)
#     plt.show()

# # loop through the validation images
# print("***************Validation images***************")
# v_labels = []
# for _ in range(len(validation_dataset.filenames)):
#     image, label = validation_dataset.next()
#     # check if the image shapes match those of resnet 50
#     v_labels.append(label)
#     print(image.shape)
#     # display the first image from the iterator
#     # plt.imshow(image[0])
#     # plt.show()
#
# # # training and validation steps
# training_steps = training_dataset.n // training_dataset.batch_size
# validation_steps = validation_dataset.n // validation_dataset.batch_size

# using functional api instead of sequential -more flexible
# inputs = keras.Input(shape=(28, 28, 3))
# x = layers.Dense(512, activation='relu', name='first_layer')(inputs)
# x = layers.Dense(256, activation='relu', name='second_layer')(x)
# x = layers.Flatten()(x)
# output = layers.Dense(10, activation='softmax')(x)
# model = keras.Model(inputs=inputs, outputs=output)
#
# print(model.summary())
#
# # configure the training part of the network
# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # logits to true if sequential
#     optimizer=keras.optimizers.Adam(lr=0.001),
#     # add from logits since there was not softmax function in the dense classification layer
#     metrics=['sparse_categorical_accuracy', 'mse'],
#
# )
# # then train the network
# model.fit(training_dataset, batch_size=32, epochs=15, verbose=2)
# print(model.summary())
# # print the tensors of each layer
# for x in model.layers:
#     print("**********************")
#     print(x.output)
#     print("**********************")
#
#
# # then evaluate the model
# model.evaluate(validation_dataset, batch_size=32, verbose=2)
input_t = keras.Input(shape=(224, 224, 3))
res_model = ResNet50(include_top=False,
                     weights="imagenet",
                     input_tensor=input_t)

for i in range(len(res_model.layers[2].get_weights()[0])):
    # print("weight is")
    # print(res_model.layers[2].get_weights()[0][i])
    print("weights")
    if (i == 1):
        #returns the first feature in the convolution with its 3 channels
        #print(res_model.layers[2].get_weights()[0][i][0])
##################################1st feature in this layer 2 within the first channel of RGB######
        #return all the positive values in the convolutional feature..in the first channel
        # x1 = np.array(res_model.layers[2].get_weights()[0][i][0]) > 0
        # print(x1[0])
        # print("all the trues in channel 1 are")
        # print(sum(x1[0]))
##############end of the channel1##################################################################

##################################1st feature in this layer 2 within the second channel of RGB######
        # return all the positive values in the convolutional feature..in the first channel
        # x1 = np.array(res_model.layers[2].get_weights()[0][i][0]) > 0
        # print(x1[1])
        # print("all the trues in channel 1 are")
        # print(sum(x1[1]))
##############end of the channel1###############################################################

        ##################################1st feature in this layer 2 within the third channel of RGB######
        # return all the positive values in the convolutional feature..in the first channel
        for x in range(len(res_model.layers[2].get_weights()[0][i])):
            x1 = np.array(res_model.layers[2].get_weights()[0][i][x]) > 0
            #print(x1[2])
            for a in range(len(x1)):

                print("all the trues in channel"+str(a+1)+" are for feature " + str(x+1))
                print(sum(x1[a]))
            ##############end of the channel1###############################################################
            #get the total number of features in the covolution
              #print("number of features in the convolution layer")
            #print(len(res_model.layers[2].get_weights()[0][i]))
    else:
        print("order:")
    # print("output shape is")
    # print(res_model.layers[2].output_shape)
    #
    # print("get the weight shape")
    # print(res_model.layers[2].get_weights()[0].shape)

# try the transfer learning model
# to_res = (224, 224)
#
# model = Sequential()
# model.add(res_model)
# model.add(layers.Flatten())
# model.add(layers.BatchNormalization())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.BatchNormalization())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.BatchNormalization())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.BatchNormalization())
# model.add(layers.Dense(2, activation='softmax'))
#
# model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
#               optimizer=optimizers.Adam(lr=3e-4),
#               metrics=['accuracy'])
# history = model.fit(training_dataset, batch_size=32, epochs=5, verbose=1)
#
#
# # create a method for prediction
# # load and prepare the image
# def load_image(filename):
#     # load the image
#     img = load_img(filename, target_size=(224, 224))
#     # convert to array
#     img = img_to_array(img)
#     # reshape into a single sample with 3 channels
#     img = img.reshape(1, 224, 224, 3)
#     # center pixel data
#     img = img.astype('float32')
#     img = img - [123.68, 116.779, 103.939]
#     return img
#
#
# print("prediction with the test data")
# model.evaluate(validation_dataset, batch_size=32, verbose=2)
#
# #do some prediction
# probability_model = Sequential([model, layers.Softmax()])
#
# print("see the results for an image")
# # load the image
# images_array=['bike_242.bmp','carsgraz_253.bmp','bike_247.bmp','bike_244.bmp'];
#
# for i in range(len(images_array)):
#
#     image1 = load_image(images_array[i])
#     predictions = probability_model.predict(image1)
#
#     print("The prediction class for:"+images_array[i]+" is:")
#     predicted_class_indices = np.argmax(predictions, axis=1)
#     training_labels = training_dataset.class_indices
#     labels = dict((v,k) for k,v in training_labels.items())
#     predictions_y = [labels[k] for k in predicted_class_indices]
#
#     print(predictions_y)
