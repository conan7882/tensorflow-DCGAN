import numpy as np 
import scipy.io
import argparse
import sys, os
import random

import tensorflow as tf 

from Models import *

def main(_):

    num_channels = FLAGS.num_channels
    image_size = 64

    model = GAN(len_random_vector = FLAGS.len_random_vector, save_model_path = FLAGS.save_model_path, save_result_path = FLAGS.save_result_path, 
      image_size = image_size, batch_size = FLAGS.batch_size,
      data_type = 'MNIST', num_channel = num_channels, flag_debug = False)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      saver.restore(sess, FLAGS.save_model_path + 'my-model-9900')
      model.test_model(sess)


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
      default= 'D:\\Qian\\TestData\\Test_FCN\\GAN\\result\\',
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
      default='D:\\GoogleDrive_Qian\\Foram\\Training\\CNN_GAN\\',
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