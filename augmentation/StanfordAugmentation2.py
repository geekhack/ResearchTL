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

csv_file = "../csv/reduced_stanford_data.csv"
filepath = "../Data2/all_training/"
new_path_prefix = '../Data2/for_augmentation/'
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    images_array = []
    for i, row in enumerate(reader):

        if i == 0:
            pass  # Skip header row
        else:
            image_name =row[0]
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
    df.to_csv('../csv/raw_augmented_stanford.csv', index=False)

#########################################AUGMENTATION GENERATION##########################
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.15,
                             zoom_range=0.1, channel_shift_range=10, horizontal_flip=True)

csv_file = "../csv/raw_augmented_stanford.csv"
image_path = "../Data2/for_augmentation"
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

            image = np.expand_dims(imageio.imread(image_path + '/' + image_name), 0)

            for x, val in zip(datagen.flow(image,
                                           save_to_dir='../Data2/processed_augmentation',
                                           save_prefix=image_id + '_x',
                                           save_format='jpg'),
                              range(20)):
                pass
################################PROCESS THE IMAGES####################################
# get the files
folder = '../Data2/processed_augmentation'

images_array = []
for i in range(6):
    files = [filename for filename in os.listdir(folder) if filename.startswith(str(i) + '_x')]
    label_name = ""
    if i == 1:
        label_name = "AM General Hummer SUV 2000"
    elif i == 2:
        label_name = "Acura RL Sedan 2012"
    elif i == 3:
        label_name = "Acura TL Sedan 2012"
    elif i == 4:
        label_name = "Acura TL Type-S 2008"
    elif i == 5:
        label_name = "Acura TSX Sedan 2012"

    for j in files:
        # save the data in a new csv file
        dic_toy = {'file': j, 'label': label_name, 'class_id': i}
        images_array.append(dic_toy)

x = []
for d in images_array:
    x.append(list(d.values()))

df = pd.DataFrame(x)
df.columns = ['file', 'label', 'class_id']
df.to_csv('../csv/processed_augmented_stanford.csv', index=False)