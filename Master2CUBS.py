#import MatchFolderCUBS as mt
import statistics as stats
from collections import defaultdict
import pandas as pd
import collections as ct
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, losses, optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# set the training dictionary
#mt.getSelectedTrainingImages()

# set the model hyper parameters
BATCH_SIZE = 64
# set the image size to fit the resnet model for lower overfitting
IMG_SIZE = (224, 224)

# create training and validation image generator object and rescale the images
train_dataset_gen = ImageDataGenerator(rescale=1. / 255.,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.2)

train_df = pd.read_csv("csv/skew_cubs_data.csv")

# generate the datasets(training and validation)
training_dataset = train_dataset_gen.flow_from_dataframe(dataframe=train_df,
                                                         directory="./Data4/cubs_images/",
                                                         x_col="file",
                                                         y_col="label",
                                                         batch_size=BATCH_SIZE,
                                                         target_size=IMG_SIZE,
                                                         validate_filenames=False,
                                                         class_mode="categorical",
                                                         subset="training")

validation_dataset = train_dataset_gen.flow_from_dataframe(dataframe=train_df,
                                                           directory="./Data4/cubs_images/",
                                                           x_col="file",
                                                           y_col="label",
                                                           batch_size=BATCH_SIZE,
                                                           target_size=IMG_SIZE,
                                                           validate_filenames=False,
                                                           class_mode="categorical",
                                                           subset="validation")

print("training items:", training_dataset.samples)

print("validation items:", validation_dataset.samples)

#
class_names = training_dataset.class_indices
# reverse the dictionary
new_dict = {}
for key, value in class_names.items():
    new_dict[value] = key

# for _ in range(len(training_dataset.filenames)):
#     image, label = training_dataset.next()
#     #items=ct.Counter(label[0]).most_common(1)[0][0]
#     # display the image from the iterator
#     plt.imshow(image[0])
#     label_name = new_dict[label[0]] # note you are only showing the first image of the batch
#     plt.title(label_name)
#     plt.show()

# get the pretrained model
model_name = resnet50
input_t = keras.Input(shape=(224, 224, 3))
model = model_name(include_top=False,
                   weights="imagenet",
                   input_tensor=input_t)

for layer in model.layers[:-2]:
    layer.trainable = False

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
t_model.add(layers.Dense(200, activation='softmax'))

t_model.compile(loss=losses.CategoricalCrossentropy(from_logits=True),
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

# do some prediction
probability_model = Sequential([t_model, layers.Softmax()])

print("see the results for an image")
# load the image
images_array = ['auklet.jpg', 'albatross.jpg'];

for i in range(len(images_array)):
    image1 = load_image(images_array[i])
    predictions = probability_model.predict(image1)

    print("The prediction class for:" + images_array[i] + " is:")
    predicted_class_indices = np.argmax(predictions, axis=1)
    training_labels = training_dataset.class_indices
    labels = dict((v, k) for k, v in training_labels.items())
    predictions_y = [labels[k] for k in predicted_class_indices]

    print(predictions_y)
