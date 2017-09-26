"""A generic module to read data."""
import numpy as np
import collections
from tensorflow.python.framework import dtypes

import scipy.io
import os.path

import tensorflow as tf
import argparse
import sys
FLAGS = None

class Dataset(object):
    def __init__(self, file_dir, file_prex, file_list, im_size, num_channel = 1,
                  one_hot = False, dtype=dtypes.float64, reshape=False):

        self._num_image = len(file_list)
        self._epochs_completed = 0
        self._index_im_in_epoch = 0
        self._index_sample_in_im = 0
        self._num_sample_in_im = 0

        self._file_list = file_list
        self._file_prex = file_prex
        self._im_size = im_size
        self._num_channel = num_channel
        self._one_hot = one_hot
        self._dtype = dtype
        self._reshape = reshape
        self._file_dir = file_dir

        self._images = []
        self._labels = []

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def set_epochs_completed(self, value):
        self._epochs_completed = value

    def next_batch(self, batch_size, batch_im_num):
        if (self._num_sample_in_im == 0):
            self.next_image(batch_im_num)
            self._index_sample_in_im = 0
        start = self._index_sample_in_im
        self._index_sample_in_im += batch_size
        if self._index_sample_in_im > self._num_sample_in_im:
            self.next_image(batch_im_num)
            # Finished epoch
            # self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_sample_in_im)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

            # Start next epoch
            start = 0
            self._index_sample_in_im = batch_size
            assert batch_size <= self._num_sample_in_im
        end = self._index_sample_in_im

        return self._images[start:end], self._labels[start:end]



    def next_image(self, batch_im_num):
        start = self._index_im_in_epoch
        self._index_im_in_epoch += batch_im_num
        if self._index_im_in_epoch > self._num_image:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_image)
            np.random.shuffle(perm)
            self._file_list = self._file_list[perm]

            # Start next epoch
            start = 0
            self._index_im_in_epoch = batch_im_num
            assert batch_im_num <= self._num_image
        end = self._index_im_in_epoch

        cur_im_data = read_by_image(self._file_dir, self._file_prex, self._file_list[start:end], 
            self._im_size, num_channel = self._num_channel, one_hot=self._one_hot, 
            dtype=self._dtype, reshape=self._reshape)

        self._images = cur_im_data.images
        self._labels = cur_im_data.labels
        self._num_sample_in_im = cur_im_data.num_examples

class ImData(object):
    """batch_data class object."""

    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float64,
                 reshape=True):
        """Initialize the class."""
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                images.shape[1] * images.shape[2])

        self._images = images
        self._num_examples = images.shape[0]
        self._labels = labels
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def shuffle_sample(self):
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

def count_data(file_dir, file_prex, file_list):
    num_image = 0
    num_samples = 0
    num_positive = 0
    num_negative = 0

    for im_id in file_list:
        file_path = file_dir + file_prex + "%02d" % (im_id) + '.mat'
        print(file_path)
        mat = scipy.io.loadmat(file_path)
        xt = mat['EdgeList']
        yt = mat['NegativeList']

        num_image += 1
        num_positive += xt.shape[0]
        num_negative += yt.shape[0]

    num_samples = num_positive + num_negative
    return (num_image, num_samples, num_positive, num_negative)

def shuffle_data(data):
    NumData = data.shape[0]
    perm = np.arange(NumData)
    np.random.shuffle(perm)
    data = data[perm]
    return data

def read_by_image(file_dir, file_prex, file_name_list, im_size, 
                  num_channel = 1, one_hot=False, dtype=dtypes.uint8, reshape=False):
    """Load data of one image."""

    positive = np.empty(shape=[0, im_size, im_size, num_channel])
    negtive = np.empty(shape=[0, im_size, im_size, num_channel])

    for im_id in file_name_list:
        file_path = file_dir + file_prex + "%02d" % (im_id) + '.mat'
            # print(file_path)
        mat = scipy.io.loadmat(file_path)
        xt = mat['EdgeList']
        yt = mat['NegativeList']
        xt = np.reshape(xt, [xt.shape[0], xt.shape[1], xt.shape[2], num_channel])
        yt = np.reshape(yt, [yt.shape[0], yt.shape[1], yt.shape[2], num_channel])

        positive = np.append(positive, xt, axis=0)
        negtive = np.append(negtive, yt, axis=0)

    positive = shuffle_data(positive)
    negtive = shuffle_data(negtive)
    num_positive = positive.shape[0]
    num_negative = negtive.shape[0]
    # print('Positive samples: {}, Negative samples: {}'.format(num_positive, num_negative))

    sample_data = np.append(positive, negtive, axis=0)
    sample_label = np.append(np.ones(shape=[num_positive,1]), np.zeros(shape=[num_negative,1]), axis=0)

    perm = np.arange(sample_data.shape[0])
    np.random.shuffle(perm)
    sample_data = sample_data[perm]
    sample_label = sample_label[perm]

    im_sample = ImData(sample_data, dense_to_one_hot(sample_label, num_classes=2), dtype=dtype, reshape=reshape)
    im_sample.shuffle_sample()

    return im_sample

def prepare_data_set(file_dir, file_prex, train_file_list, validation_file_list, im_size, num_channel = 1, one_hot=False,
                         dtype=dtypes.uint8, reshape=False):
    train_num_image, train_num_samples, train_num_positive, train_num_negative = count_data(file_dir, file_prex, train_file_list)
    validation_num_image, validation_num_samples, validation_num_positive, validation_num_negative = count_data(file_dir, file_prex, validation_file_list)

    print('Training Data: image [{}], samples [{}], positive [{}], negtive [{}]'
            .format(train_num_image, train_num_samples, train_num_positive, train_num_negative))
    print('validation Data: image [{}], samples [{}], positive [{}], negtive [{}]'
            .format(validation_num_image, validation_num_samples, validation_num_positive, validation_num_negative))

    train = Dataset(file_dir, file_prex, train_file_list, im_size, num_channel = num_channel,
        one_hot = one_hot, dtype=dtype, reshape=reshape)

    validation = Dataset(file_dir, file_prex, validation_file_list, im_size, num_channel = num_channel,
        one_hot = one_hot, dtype=dtype, reshape=reshape)

    ds = collections.namedtuple('Datasets', ['train', 'validation'])
    return ds(train = train, validation = validation)


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[[index_offset + labels_dense.ravel()]] = 1
    return labels_one_hot

def main(_):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='E:\\course\\TAECE792\\Project\\project1\\TrainingTensorFlow\\Training\\',
        help='Directory for storing training data'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)