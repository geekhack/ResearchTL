import MatcherFolder as mt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# set the training dictionary
mt.getSelectedTrainingImages()

# set the validation dictionary
mt.getSelectedValidationImages()

# set the model hyper parameters
BATCH_SIZE = 32
# set the image size to fit the resnet model for lower overfitting
IMG_SIZE = (28, 28)

# create training and validation image generator object and rescale the images
train_dataset_gen = ImageDataGenerator(rescale=1. / 255.)
validation_dataset_gen = ImageDataGenerator(rescale=1. / 255.)

train_df = pd.read_csv("train_data.csv")
validation_df = pd.read_csv("validation_data.csv")

# data(training and validation) preprocess - show the first 3 images
print("*********training data****************")
print(train_df.head(3))

print("*********validation data**************")
print(validation_df.head())

# generate the datasets(training and validation)
training_dataset = train_dataset_gen.flow_from_dataframe(dataframe=train_df,
                                                         directory="./Data/train/",
                                                         x_col="file",
                                                         y_col="label",
                                                         shuffle=True,
                                                         batch_size=BATCH_SIZE,
                                                         target_size=IMG_SIZE,
                                                         class_mode="categorical",
                                                         subset="training")

validation_dataset = validation_dataset_gen.flow_from_dataframe(dataframe=validation_df,
                                                         directory="./Data/validate/",
                                                         x_col="file",
                                                         y_col="label",
                                                         shuffle=True,
                                                         batch_size=BATCH_SIZE,
                                                         target_size=IMG_SIZE,
                                                         class_mode="categorical",
                                                         subset=None)
# show the classes for both training and validation sets
print(training_dataset.class_indices)
print(validation_dataset.class_indices)

# show the images filenames
print("*************training images in dataset****************")
print(training_dataset.filenames)
print("*************validation images in dataset**************")
print(validation_dataset.filenames)

# loop through the training images

print("***************Training images***************")
for _ in range(3):
    image, label = training_dataset.next()
    # check if the image shapes match those of resnet 50
    print(image.shape)
    # display the first image from the iterator
    # plt.imshow(image[0])
    # plt.show()

# loop through the validation images
print("***************Validation images***************")
for _ in range(3):
    image, label = validation_dataset.next()
    # check if the image shapes match those of resnet 50
    print(image.shape)
    # display the first image from the iterator
    # plt.imshow(image[0])
    # plt.show()
#
# # training and validation steps
training_steps = training_dataset.n // training_dataset.batch_size
validation_steps = validation_dataset.n // validation_dataset.batch_size
