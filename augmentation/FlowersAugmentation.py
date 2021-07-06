import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import csv
import pandas as pd
import numpy as np
import scipy
from scipy import ndimage
import imageio

# # set the training dictionary
# # mt.getSelectedTrainingImages()

csv_file = "../../csv/reduced_flowers_data.csv"
filepath = "../../Data3/train/"
new_path_prefix = '../../Data3/for_augmentation/'
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    images_array = []
    for i, row in enumerate(reader):

        if i == 0:
            pass  # Skip header row
        else:
            image_name1 =row[0]
            image_name2 = str(image_name1).split('/')
            image_name = image_name2[1]
            image_class =row[1]
            image_id =row[2]

            new_filename = os.path.join(new_path_prefix, row[0])
            old_filename = os.path.join(filepath, row[0])
            shutil.copy(old_filename, new_filename)
            #create a new csv with the data
            dic_toy = {'file': image_name, 'label': image_class,'class_id':image_id}
            images_array.append(dic_toy)

    x = []
    for d in images_array:
        x.append(list(d.values()))

    df = pd.DataFrame(x)
    df.columns = ['file', 'label','class_id']
    df.to_csv('../../csv/raw_augmented_flowers.csv', index=False)

# #########################################AUGMENTATION GENERATION##########################
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.15,
                             zoom_range=0.1, channel_shift_range=10, horizontal_flip=True)

csv_file = "../../csv/raw_augmented_flowers.csv"

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    images_array = []
    for i, row in enumerate(reader):

        if i == 0:
            pass  # Skip header row
        else:
            image_name = row[0]
            image_class = row[1]
            image_id = row[2]
            image_path = "../../Data3/for_augmentation/"+image_class
            image = np.expand_dims(imageio.imread(image_path + '/' + image_name), 0)

            for x, val in zip(datagen.flow(image,
                                           save_to_dir='../../Data3/processed_augmentation/'+image_class,
                                           save_prefix=image_id + '_x',
                                           save_format='jpg'),
                              range(20)):
                pass
################################PROCESS THE IMAGES####################################
# get the files
folder = '../../Data3/processed_augmentation'
images_array = []
for x in os.listdir(folder):
    for i in range(6):

        files = [filename for filename in os.listdir(folder+'/'+x) if filename.startswith(str(i) + '_x')]
        label_name = x

        for j in files:
            # save the data in a new csv file
            dic_toy = {'file':x+'/'+j, 'label': label_name, 'class_id': i}
            images_array.append(dic_toy)

    x = []
    for d in images_array:
        x.append(list(d.values()))

df = pd.DataFrame(x)
df.columns = ['file', 'label', 'class_id']
df.to_csv('../../csv/processed_augmented_flowers.csv', index=False)



