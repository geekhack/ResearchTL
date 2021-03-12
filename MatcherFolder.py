import cv2
import numpy as np
from os import listdir
import tensorflow as tf
from os.path import isfile, join
import json

try:
    to_unicode = np.unicode
except NameError:
    to_unicode = str

selected_images_array = []
good = []


# merge the images from the various classes into their own dictionary lists
def merge_dictionary_lists(dict_list):
    return {
        k: [d.get(k) for d in dict_list if k in d]  # explanation A
        for k in set().union(*dict_list)  # explanation B
    }


# save the filenames in an array for use when training
def saveTrainingFileNames(img_list):
    imgs_lst = merge_dictionary_lists(img_list)
    upd_dictionary = {}
    # write the specific details into own file
    for mg in imgs_lst:
        # # create dictionary with a set(remove duplicates)for storing the data
        selected_images = {mg: list(set(imgs_lst[mg]))}
        upd_dictionary.update(selected_images)

    # write to the json file with the updated dictionary
    with open('selectedimages.json', 'w', encoding='utf8') as outfile:
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


# setup the delete functionality of details from the json file###################################
def deleteFileContents():
    # in case the json file has some data,clear it to accomodate the data acquired from the new reading
    with open('selectedimages.json', 'w', encoding='utf8') as json_file:
        json_file.truncate()
        json_file.close()
    # end of clearing the file contents


##############################end of delete###################################################
# the selected image to be passed into the filter is the one coming from the source dataset into the
# target domain.This will filter the images closer to the source domain from the target hence improving
# their chance of being used in the experiment(or to give some higher accuracy) hence reducing the
# overfitting

#############################setup the training function #######################################################
def imageProcessing(query_image, training_image, imageName, xx, label):
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

    # check some matching of the two images
    matcher = cv2.BFMatcher()
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

    for m, j in image_match:

        if m.distance < lowest_ratio * j.distance:
            good.append([m])

    if len(good) > 220:
        image_label = {label: imageName}
        resultMsg = 'there are %d good matches ' % (len(good)) + 'for image ' + imageName + ' with for ' + xx + 'for ' \
                                                                                                                'label:' + label
        # print(resultMsg)
        return image_label


######################################read the images to feature match from folder####################################################
# use the images from resnet(source) to feature match those for others(in Data folder
# e.g. pests,dogs and cats etc
def sortTrainImages():
    # set the parameters for the training data
    # get the class labels from training datasets
    p = []
    data_path = 'D:/pycharmProjects/ResearchTL/ResearchTL/Data'
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=20)
    labels = img_gen.flow_from_directory(data_path + '/train')
    train_labels = labels.class_indices.keys()

    for lbl in train_labels:
        my_path = 'D:/pycharmProjects/ResearchTL/ResearchTL/Data/train/' + lbl  # images from the imagenet source
        only_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
        images = np.empty(len(only_files), dtype=object)

        # set the parameters for the matching feature images
        resnet_path = 'D:/pycharmProjects/ResearchTL/ResearchTL/RefImages'
        p_files = [f for f in listdir(resnet_path) if isfile(join(resnet_path, f))]
        pests_images = np.empty(len(p_files), dtype=object)

        for m in range(0, len(p_files)):
            deleteFileContents()
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
                        p.append(imageProcessing(query_img, images[n], imageName, p_files[s], lbl))

                break
    return p


#######################################end of reading images from pests folder################################################
# then call the save method
# get the class items
train_images = sortTrainImages()
# new_selected_images = train_images[1:]
new_selected_images = train_images

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

    with open('selectedimages.json') as selected_images_file:
        s_data = json.load(selected_images_file)
        s_images = s_data

    return s_images
