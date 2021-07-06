import cv2
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
        dict_v_array = []
        item1 = str(json.dumps(index))
        split_item = item1.split(":")
        if len(split_item) == 3:
            key_item = split_item[0]
            key_1 = key_item.replace("{", "")
            key_2 = key_1.replace('"', "")
            key_value = split_item[2]
            key_3 = key_value.replace("}", "")
            key_4 = key_3.replace('"', "")

            key_5 = split_item[1].replace('"', "")

            dict_v_array.append((key_4, key_5))
            # dict_v_array.append(key_5)
            selected_images = {key_2: dict_v_array}

        upd_dictionary.update(selected_images)

    # write to the json file with the updated dictionary
    with open('json/selected_insects_training_images.json', 'w', encoding='utf8') as outfile:
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
    with open('json/selected_insects_training_images.json', 'w', encoding='utf-8') as json_file:
        json_file.truncate()
        json_file.close()
    # end of clearing the file contents


# the selected image to be passed into the filter is the one coming from the source dataset into the
# target domain.This will filter the images closer to the source domain from the target hence improving
# their chance of being used in the experiment(or to give some higher accuracy) hence reducing the
# overfitting

#############################setup the training function #######################################################
def imageProcessing(query_image, training_image, imageName, xx, label, index_label):
    # create a dictionary to return image and its label
    image_label = {}

    # train_img = cv2.imread(training_image)
    # Convert it to grayscale
    # print(tr_image_name)
    query_img_bw = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector algorithm
    orb = cv2.ORB_create()
    # Now detect the keypoints and compute descriptors on both images
    qImageKeypoints, qImageDescriptors = orb.detectAndCompute(query_img_bw, None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

    if trainDescriptors is None:
        return False
    else:
        # check some matching of the two images
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.match(qImageDescriptors, trainDescriptors)

        # draw the matches to the final image
        final_img = cv2.drawMatches(query_image, qImageKeypoints,
                                    training_image, trainKeypoints, matches[:20], None)
        # resize the final image
        cv2.resize(final_img, (1000, 650))

        # get the match percentage
        image_match = matcher.knnMatch(qImageDescriptors, trainDescriptors, k=2)
        # Apply ratio test
        lowest_ratio = 0.89

        for i, pair in enumerate(image_match):
            try:
                m, j = pair
                if m.distance < lowest_ratio * j.distance:
                    good.append([m])
            except ValueError:
                pass

        if len(good) > 950:
            image_label = {label + ":" + str(index_label): imageName}
            resultMsg = 'there are %d good matches ' % (
                len(good)) + 'for image ' + imageName + ' with for ' + xx + 'for ' \
                                                                            'label:' + label
            print(resultMsg)
            return image_label


######################################read the images to feature match from folder####################################################
# use the images from resnet(source) to feature match those for others(in Data folder
# e.g. pests,dogs and cats etc
def sortTrainImages():
    # set the parameters for the training data
    # get the class labels from training datasets
    p = []
    # data_path = 'D:/pycharmProjects/ResearchTL/ResearchTL/Data'
    data_path = './Data6'
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=20)
    labels = img_gen.flow_from_directory(data_path + '/all_images')

    # convert a csv into a list
    all_insects_df = pd.read_csv("csv/merged_insects.csv")

    training_list = list()

    for j in all_insects_df.index:
        image = all_insects_df['file'][j]
        label = all_insects_df['label'][j]
        class_id = all_insects_df['class_id'][j]
        training_list.append((image, label, class_id))

    my_path = './Data6/all_images'  # images from the imagenet source
    only_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
    images = np.empty(len(only_files), dtype=object)

    # set the parameters for the matching feature images
    resnet_path = './Refinsects'
    p_files = [f for f in listdir(resnet_path) if isfile(join(resnet_path, f))]
    pests_images = np.empty(len(p_files), dtype=object)

    for m in range(0, len(p_files)):
        deleteTrainingFileContents()
        ##############read all the other images from the folder##################################
        pests_images[m] = cv2.imread(join(resnet_path, p_files[m]))
        # get the name of the image
        imageName_x = p_files[m]
        # execute the loop once to avoid execution of the outer loop in the inner loop
        if m == 1:
            for s in range(0, len(p_files)):

                for n in range(0, len(only_files)):
                    query_img = cv2.imread(resnet_path + '/' + p_files[s])
                    # ##############read all the other images from the folder##################################
                    images[n] = cv2.imread(join(my_path, only_files[n]))
                    # get the name of the image
                    imageName = only_files[n]
                    # then perform some orb on the image at position n
                    # get the label of this image here
                    for c in training_list:
                        if c[0] == imageName:
                            # print("jina:", c[2])
                            p.append(imageProcessing(query_img, images[n], imageName, p_files[s], c[1], c[2]))

            break
    return p


#######################################end of reading images from training folder################################################


# then call the save method
# get the class items
new_selected_images = sortTrainImages()

updated_new_selected_images = []
for val in new_selected_images:
    if val != None:
        # break the val into 2 to get the image and the label
        updated_new_selected_images.append(val)

saveTrainingFileNames(updated_new_selected_images)


##################################end of writing images into json file############################################

# once the images have been written,read them and develop a dataset for use in the model##########################
# first display the data in the dictionary
# load the json file

def getSelectedTrainingImages():
    with open('json/selected_insects_training_images.json') as selected_images_file:
        s_data = json.load(selected_images_file)
        images_array = []
        for o in s_data:
            for p in s_data[o]:
                dic_toy = {'label': o, 'file': p[0], 'class_id': p[1]}
                images_array.append(dic_toy)

        x = []
        for d in images_array:
            x.append(list(d.values()))

        df = pd.DataFrame(x)
        df.columns = ['label', 'file', 'class_id']
        df.to_csv('csv/insects_train_data.csv', index=False)


# once the images have been written,read them and develop a dataset for use in the model##########################
# first display the data in the dictionary
# load the json file
