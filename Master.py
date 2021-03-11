import Matching as mt
import tensorflow as tf
import DataSetCreator
import numpy as np
from tensorflow.keras.preprocessing import image as Img

data_path = 'Data'
# get the images for loading from the json file
sel_images = mt.getSelectedImages()
# set the ratios for training and validation
# 70% of the images to be training
training_len = int(0.70 * len(sel_images))
#convert into a numpy array
numpy_array=np.array(sel_images)
#divide the list into two
training_array, validation_array = np.split(np.random.permutation(numpy_array),[training_len])


# loop through the training dataset and display
print("training set::")
train_list_dataset = []
for tr_img in training_array:
    tr_image_load = Img.load_img(data_path + "/" + tr_img)
    train_list_dataset.append(data_path + "/" + tr_img)

print(train_list_dataset)

# loop through the validation dataset and display
print("validation set::")
val_list_dataset = []

for vl_img in validation_array:
    vl_image_load = Img.load_img(data_path + "/" + vl_img)
    val_list_dataset.append(data_path + "/" + vl_img)

print(val_list_dataset)

dataset_check = tf.data.Dataset.list_files(data_path+"/*.jpg")

for f in dataset_check:
    item = f.numpy()
    #print(item)

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)
images, labels = next(img_gen.flow_from_directory(data_path+'/train'))
print(images[0])
# dataProcessor = DataSetCreator.DataSetCreator(32, 300, 500,dataset_check)
# dataProcessor.load_process()
#
#
# image_batch, label_batch = dataProcessor.get_batch()

