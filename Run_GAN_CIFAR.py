import numpy as np 
import scipy.io
import argparse
import sys, os
import random
from PIL import Image as pimg

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

from Models import *
import data_image 

def main(_):

    num_channels = FLAGS.num_channels
    image_size = 32

    model = GAN(len_random_vector = FLAGS.len_random_vector, save_model_path = FLAGS.save_model_path, save_result_path = FLAGS.save_result_path, 
      image_size = image_size, batch_size = FLAGS.batch_size,
      data_type = 'MNIST', num_channel = num_channels, flag_debug = False)

    saver = tf.train.Saver()
    
    writer = tf.summary.FileWriter(FLAGS.save_model_path)


    training_data = data_image.CIFAT10('D:\\Qian\\GitHub\\workspace\\tensorflow-DCGAN\\cifar-10-python.tar\\')

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      writer.add_graph(sess.graph)

      batch_size = FLAGS.batch_size
      step = 0
      while training_data.epochs_completed < 100:
      # for step in range(10000):
        # batch, batch_ys = all_data.train.next_batch(batch_size, 1)
        # batch = np.reshape(batch, [len(batch), image_size, image_size, 1])
        batch = training_data.next_batch(batch_size)/255.0*2.0-1.0

        # model.train_model(batch, step, 100, saver, sess, writer)
        model.train_model(batch, step, 0, training_data.epochs_completed, 100, saver, sess, writer)
        step += 1

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
      default=3,
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
      default=64,
      help='Size of batch'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)