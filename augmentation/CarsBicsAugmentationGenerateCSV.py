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
################################PROCESS THE IMAGES####################################
# get the files
folder = '../Data/train'
images_array = []
for x in os.listdir(folder):
    for i in range(2):

        files = [filename for filename in os.listdir(folder+'/'+x)]
        label_name = x

        for j in files:
            print(j)
            # save the data in a new csv file
            dic_toy = {'file':j, 'label': label_name, 'class_id': i}
            images_array.append(dic_toy)

y = []
for d in images_array:
   y.append(list(d.values()))

df = pd.DataFrame(y)
df.columns = ['file', 'label', 'class_id']
df.to_csv('../csv/raw_car_bics.csv', index=False)



