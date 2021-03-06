import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from tkinter.filedialog import askopenfilename
import json

try:
    to_unicode = np.unicode
except NameError:
    to_unicode = str

# set the parameters
my_path = 'data'
only_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
images = np.empty(len(only_files), dtype=object)
selected_images_array = []
good = []


# save the filenames in an array for use when training
def saveFileNames(img_list):
    # create dictionary for storing the data
    selected_images = {'target_images': img_list}

    with open('selectedimages.json', 'w', encoding='utf8') as outfile:
        str_ = json.dumps(selected_images,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))


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
def imageProcessing(query_image, training_image):
    train_img = cv2.imread(training_image)
    # Convert it to grayscale
    query_img_bw = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector algorithm
    orb = cv2.ORB_create()
    # Now detect the keypoints and compute descriptors on both images
    qImageKeypoints, qImageDescriptors = orb.detectAndCompute(query_img_bw, None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

    # check some matching of the two images
    matcher = cv2.BFMatcher()
    matches = matcher.match(qImageDescriptors, trainDescriptors)

    # draw the matches to the final image
    final_img = cv2.drawMatches(query_img, qImageKeypoints,
                                train_img, trainKeypoints, matches[:20], None)
    # resize the final image
    cv2.resize(final_img, (1000, 650))

    # get the match percentage
    image_match = matcher.knnMatch(qImageDescriptors, trainDescriptors, k=2)

    # Apply ratio test
    lowest_ratio = 0.89

    for m, j in image_match:

        if m.distance < lowest_ratio * j.distance:
            good.append([m])
    if len(good) > 52:
        resultMsg = 'there are %d good matches ' % (len(good)) + 'for image ' + training_image
        return training_image


############################################## end of training ##########################################################


######################################read the image to be uploaded####################################################
# image to compare with
imageq = askopenfilename(filetypes=[("image", "*.jpg")])  # queryImage
query_img = cv2.imread(imageq)

#######################################end of reading image for upload################################################


################################loop through the images##############################################################
p = []
for n in range(0, len(only_files)):
    deleteFileContents()
    ##############read all the other images from the folder##################################
    images[n] = cv2.imread(join(my_path, only_files[n]))
    # get the name of the image
    imageName = only_files[n]
    # then perform some orb on the image at position n
    p.append(imageProcessing(query_img, imageName))

# then call the save method
new_selected_images = p[1:]
saveFileNames(new_selected_images)


##################################end of writing images into json file############################################

# once the images have been written,read them and develop a dataset for use in the model##########################
# first display the data in the dictionary
# load the json file

def getSelectedImages():
    with open('selectedimages.json') as selected_images_file:
        s_data = json.load(selected_images_file)
        for a in s_data['target_images']:
            print(a + " iko kwa folder")