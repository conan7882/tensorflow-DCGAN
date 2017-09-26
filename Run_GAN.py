import numpy as np 
import scipy.io
import argparse
import sys, os
import random
from PIL import Image as pimg

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

from Models import *
import dataset
import data_image 

def main(_):

    num_channels = FLAGS.num_channels
    image_size = 64

    model = GAN(len_random_vector = FLAGS.len_random_vector, save_model_path = FLAGS.save_model_path, save_result_path = FLAGS.save_result_path, 
      image_size = image_size, batch_size = FLAGS.batch_size, data_type = 'MNIST', flag_debug = False)

    saver = tf.train.Saver()
    
    writer = tf.summary.FileWriter(FLAGS.save_model_path)

    # num_labeled_image, num_train_im = 2, 1
    # num_validation_im = num_labeled_image - num_train_im
    # perm = np.arange(num_labeled_image) + 1
    # all_data = dataset.prepare_data_set(FLAGS.train_dir, 'Training_', 
    #                                     perm[num_validation_im:], perm[0:num_validation_im], image_size, 
    #                                     num_channel = num_channels, one_hot=True, reshape=False)

    training_data = data_image.prepare_data_set(FLAGS.train_dir, 0.05, num_channels = num_channels, isSubstractMean = False)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      writer.add_graph(sess.graph)

      batch_size = FLAGS.batch_size
      for step in range(10000):
        # batch, batch_ys = all_data.train.next_batch(batch_size, 1)
        # batch = np.reshape(batch, [len(batch), image_size, image_size, 1])
        batch = training_data.train.next_batch(batch_size)
        batch = batch*2.0 - 1.0
        model.train_model(batch, step, 0, training_data.train.epochs_completed, 100, saver, sess, writer)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--save_model_path',
      type=str,
      default= 'D:\\Qian\\TestData\\Test_FCN\\Trained_FCN\\GAN_prob\\',
      help='Directory for storing trained parameters'
  )
  parser.add_argument(
      '--save_result_path',
      type=str,
      default= 'D:\\Qian\\TestData\\Test_FCN\\GAN\\',
      help='Directory for storing result'
  )

  parser.add_argument(
      '--num_channels',
      type=int,
      default=1,
      help='Number of input channel'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='D:\\GoogleDrive_Qian\\Foram\\Training\\CNN_GAN_ORIGINAL_64\\',
      help='Directory for storing training data'
  )

  parser.add_argument(
      '--len_random_vector',
      type=int,
      default=100,
      help='Length of input random vector'
  )

  parser.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='Size of batch'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)