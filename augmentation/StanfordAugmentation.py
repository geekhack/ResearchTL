import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import csv

# set the training dictionary
# mt.getSelectedTrainingImages()

csv_file = "../csv/reduced_stanford_data.csv"
filepath = "../Data2/all_training/"
new_path_prefix = '../Data2/for_augmentation/'
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):

        if i == 0:
            pass  # Skip header row
        else:
            print(row[0])
            new_filename = os.path.join(new_path_prefix, row[0])
            old_filename = os.path.join(filepath, row[0])
            shutil.copy(old_filename, new_filename)


data_directory='../Data2/for_augmentation/'
processed_directory='../Data2/processed_augmentation'


datagen = ImageDataGenerator(
    rotation_range=45,  # Random rotation between 0 and 45
    width_shift_range=0.2,  # % shift
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant', cval=125)  # Also try nearest, constant, reflect, wrap

i = 0
for batch in datagen.flow_from_directory(directory=data_directory,
                                         batch_size=16,
                                         target_size=(256, 256),
                                         color_mode="rgb",
                                         save_to_dir=processed_directory,
                                         save_prefix='aug',
                                         save_format='jpeg'):
    print(batch.class_indices)
    i += 1
    if i > 2:
        break
