import pandas as pd
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import os.path
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2 as l2
from tensorflow.keras import layers, losses, optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

BATCH_SIZE = 32
# set the image size to fit the resnet model for lower overfitting
IMG_SIZE = (300, 300)

train_dataset_gen = ImageDataGenerator(rescale=1. / 255.,
                                       # shear_range=0.2,
                                       # zoom_range=0.2,
                                       # horizontal_flip=True,
                                       # validation_split=0.2
                                       )
train_df = pd.read_csv("skew_cubs_data.csv")

training_dataset = train_dataset_gen.flow_from_dataframe(dataframe=train_df,
                                                         directory="../cubs_images/",
                                                         shuffle=True,
                                                         x_col="file",
                                                         y_col="label",
                                                         batch_size=BATCH_SIZE,
                                                         target_size=IMG_SIZE,
                                                         validate_filenames=False,
                                                         class_mode="categorical",
                                                         subset="training")
model_name = resnet50

# load model without classifier layers
model = model_name(include_top=False, input_shape=(300, 300, 3))
for layer in model.layers:
    layer.trainable = False
# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
flat1 = Dropout(0.5)(flat1)
class1 = Dense(1024, activation='relu')(flat1)
class1 = Dropout(0.5)(class1)
output = Dense(200, activation='softmax')(class1)
# define new model
t_model = Model(inputs=model.inputs, outputs=output)

t_model.compile(loss=losses.CategoricalCrossentropy(from_logits=True),
                optimizer=optimizers.Adam(lr=3e-4),
                metrics=['accuracy'])
# improves history = t_model.fit(training_dataset, batch_size=32, epochs=100, verbose=1, callbacks=[callback])

history = t_model.fit(training_dataset, batch_size=32, epochs=30, verbose=1)


def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(300, 300))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 300, 300, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img


# do some prediction
probability_model = Sequential([t_model, layers.Softmax()])

print("see the results for an image")
# load the image
images_array = ['Laysan_Albatross_0068_726.jpg', 'Bobolink_0047_9204.jpg', 'Black_Footed_Albatross_0031_100'];

for i in range(len(images_array)):
    image1 = load_image(images_array[i])
    predictions = probability_model.predict(image1)

    print("The prediction class for:" + images_array[i] + " is:")
    predicted_class_indices = np.argmax(predictions, axis=1)
    training_labels = training_dataset.class_indices
    labels = dict((v, k) for k, v in training_labels.items())
    predictions_y = [labels[k] for k in predicted_class_indices]

    print(predictions_y)
