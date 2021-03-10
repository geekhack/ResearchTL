import tensorflow as tf
import os

CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'] # flower labels (folder names in the data)

class DataSetCreator(object):
    def __init__(self, batch_size, image_height, image_width, dataset):
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.dataset = dataset

    def _get_class(self, path):
        pat_splited = tf.strings.split(path, os.path.sep)
        return pat_splited[-2] == CLASS_NAMES

    def _load_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return tf.image.resize(image, [self.image_height, self.image_width])

    def _load_labeled_data(self, path):
        label = self._get_class(path)
        image = self._load_image(path)
        return image, label

    def load_process(self, shuffle_size=1000):
        self.loaded_dataset = self.dataset.map(self._load_labeled_data,
                                               num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.loaded_dataset = self.loaded_dataset.cache()

        # Shuffle data and create batches
        self.loaded_dataset = self.loaded_dataset.shuffle(buffer_size=shuffle_size)
        self.loaded_dataset = self.loaded_dataset.repeat()
        self.loaded_dataset = self.loaded_dataset.batch(self.batch_size)

        # Make dataset fetch batches in the background during the training of the model.
        self.loaded_dataset = self.loaded_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def get_batch(self):
        return next(iter(self.loaded_dataset))
