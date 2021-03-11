import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from tkinter.filedialog import askopenfilename
import json

pests_path = 'D:/pycharmProjects/ResearchTL/ResearchTL/Data/train/bics'
p_files = [f for f in listdir(pests_path) if isfile(join(pests_path, f))]
pests_images = np.empty(len(p_files), dtype=object)

p = []
for n in range(0, len(p_files)):
    ##############read all the other images from the folder##################################

    pests_images[n] = cv2.imread(join(pests_path, p_files[n]))
    print(pests_images[n])
    # get the name of the image
    imageName = p_files[n]
    # then perform some orb on the image at position n
    p.append(imageName)

