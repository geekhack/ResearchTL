import numpy as np
from os import listdir
import tensorflow as tf
from os.path import isfile, join
import json
import pandas as pd
from scipy.io import loadmat

try:
    to_unicode = np.unicode
except NameError:
    to_unicode = str

selected_images_array = []
good = []


# merge the images from the various classes into their own dictionary lists
def merge_dictionary_lists(dictionary_list):
    return {
        i: [dictionary.get(i) for dictionary in dictionary_list if i in dictionary]
        for i in set().union(*dictionary_list)
    }


# save the filenames in an array for use when training
def saveTrainingFileNames(img_list):
    ig = iter(img_list)
    upd_dictionary = {}

    # write the specific details into own file
    for index in ig:
        item1 = str(json.dumps(index))
        split_item = item1.split(":")
        dict_v_array = []
        if len(split_item) == 3:
            key_item = split_item[0]
            key_1 = key_item.replace("{", "")
            key_2 = key_1.replace('"', "")

            key_value = split_item[2]
            key_3 = key_value.replace("}", "")
            key_4 = key_3.replace('"', "").strip()

            key_5 = split_item[1].replace('"', "").strip()

            dict_v_array.append((key_2, key_5))
            selected_images = {key_4: dict_v_array}
            upd_dictionary.update(selected_images)

    print(upd_dictionary)
    # write to the json file with the updated dictionary
    with open('../json/skew_stanford_training_images.json', 'w', encoding='utf8') as outfile:
        str_ = json.dumps(cleanDictionary(upd_dictionary),
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)

        outfile.write(to_unicode(str_))


# clean up the dictionary if null items exist
def cleanDictionary(dictry):
    better_dict = {}
    for x, y in dictry.items():
        if isinstance(y, dict):
            nested = cleanDictionary(y)
            if len(nested.keys()) > 0:
                better_dict[x] = nested
        elif y is not None:
            better_dict[x] = y
    return better_dict


# setup the delete functionality of details from the json training file###################################
def deleteTrainingFileContents():
    # in case the json file has some data,clear it to accomodate the data acquired from the new reading
    with open('../json/skew_stanford_training_images.json', 'w', encoding='utf-8') as json_file:
        json_file.truncate()
        json_file.close()
    # end of clearing the file contents


#############################setup the training function #######################################################
def imageProcessing(imageName, label, lbl_id):
    image_label = {label + ":" + str(lbl_id): imageName}
    resultMsg = 'there are %d good matches ' % (len(good)) + 'for image ' + imageName + ' with for ' + 'for ' \
                                                                                                       'label:' + label

    return image_label


######################################read the images to feature match from folder####################################################
# use the images from resnet(source) to feature match those for others(in Data folder
# e.g. pests,dogs and cats etc
def sortTrainImages():
    # set the parameters for the training data
    # get the class labels from training datasets
    p = []
    # data_path = 'D:/pycharmProjects/ResearchTL/ResearchTL/Data'
    data_path = '../Data2/all_training'
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=20)
    labels = img_gen.flow_from_directory(data_path)
    # get the labels

    mat_train = loadmat('../matfiles/stanford/cars_train_annos.mat')
    meta = loadmat('../matfiles/stanford/cars_meta.mat')

    # create a dictionary to represent labels and their corresponding values

    training_labels = list()
    label_id = 0
    for l in meta['class_names'][0]:
        label_id = label_id + 1
        training_labels.append((l[0], label_id))

    training_list = list()

    for example in mat_train['annotations'][0]:
        label_x = training_labels[example[-2][0][0] - 1]
        print(example[-1][0])
        label = label_x[0]
        class_id = label_x[1]
        image = example[-1][0]
        training_list.append((image, label, class_id))

    my_path = '../Data2/all_training'  # images from the imagenet source
    only_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

    for n in range(len(only_files)):
        for x in training_list:
            # get the name of the image
            imageName = only_files[n]
            if x[0] == imageName:
                lble = x[1]
                lble_id = x[2]
                # then perform some orb on the image at position n
                p.append(imageProcessing(imageName, lble, lble_id))
    return p


#######################################end of reading images from training folder################################################


# then call the save method
# get the class items
new_selected_images = sortTrainImages()

updated_new_selected_images = []
for val in new_selected_images:
    if val != None:
        updated_new_selected_images.append(val)

saveTrainingFileNames(updated_new_selected_images)


##################################end of writing images into json file############################################

# once the images have been written,read them and develop a dataset for use in the model##########################
# first display the data in the dictionary
# load the json file

def getSelectedTrainingImages():
    with open('../json/skew_stanford_training_images.json') as selected_images_file:
        s_data = json.load(selected_images_file)
        images_array = []
        for o in s_data:
            for p in s_data[o]:
                # get the values after the / for the file
                dic_toy = {'file': o, 'label': p[0], 'class_id': p[1]}
                images_array.append(dic_toy)

        x = []
        for d in images_array:
            x.append(list(d.values()))

        df = pd.DataFrame(x)
        df.columns = ['file', 'label', 'class_id']
        df.to_csv('../csv/skew_stanford_training_data.csv', index=False)


# once the images have been written,read them and develop a dataset for use in the model##########################
# first display the data in the dictionary
# load the json file
getSelectedTrainingImages();
