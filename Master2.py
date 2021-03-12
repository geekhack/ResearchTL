import MatcherFolder as mt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# set the training dictionary
mt.getSelectedTrainingImages()

# set the model hyper parameters
BATCH_SIZE = 32
# set the image size to fit the resnet model for lower overfitting
IMG_SIZE = (28, 28)

# create image generator object and rescale the images
train_dataset_gen = ImageDataGenerator(rescale=1. / 255.)

train_df_2 = pd.read_csv("data.csv")
# data preprocess - show the first 3 images
print(train_df_2.head(3))
train_generator = train_dataset_gen.flow_from_dataframe(dataframe=train_df_2,
                                                        directory="./Data/train/",
                                                        x_col="file",
                                                        y_col="label",
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        target_size=IMG_SIZE,
                                                        class_mode="categorical",
                                                        subset="training")
# show the classes
print(train_generator.class_indices)

# show the images filenames
print(train_generator.filenames)

# loop through the images
for _ in range(3):
    image, label = train_generator.next()
    # check if the image shapes match those of resnet 50
    print(image.shape)
    # display the first image from the iterator
    plt.imshow(image[0])
    plt.show()

# training and validation steps
training_steps = train_generator.n // train_generator.batch_size
