import numpy as np 
import scipy.io
import argparse
import sys, os
import random
from PIL import Image as pimg

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

from Models import *

def main(_):

    num_channels = FLAGS.num_channels
    image_size = 28

    model = GAN(len_random_vector = FLAGS.len_random_vector, save_model_path = FLAGS.save_model_path, save_result_path = FLAGS.save_result_path, 
      image_size = image_size, batch_size = FLAGS.batch_size, data_type = 'MNIST',  flag_debug = False)

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(FLAGS.save_model_path)

    mnist_data = input_data.read_data_sets('../workspace/tensorflow-DCGAN/MNIST_data', one_hot=True)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      writer.add_graph(sess.graph)

      digit = 6
      train_digits_of_interest = []
      for image, label in zip(mnist_data.train.images, mnist_data.train.labels):
        image = image*2.-1.
          # if label[digit]:
        train_digits_of_interest.append(image)
      for image, label in zip(mnist_data.test.images, mnist_data.test.labels):
          # if label[digit] or label[digit_2]:
          image = image*2.-1.
          train_digits_of_interest.append(image)
      random.shuffle(train_digits_of_interest)

      batch_size = FLAGS.batch_size
      epoch_id = 1
      step = 0
      idx = 0
      while epoch_id <= 100:
        batch_index = step * batch_size % len(train_digits_of_interest)
        if batch_index > len(train_digits_of_interest) - batch_size:
          batch_index = len(train_digits_of_interest) - batch_size
          random.shuffle(train_digits_of_interest)
          idx = 0
          epoch_id += 1
        idx += 1
        batch = train_digits_of_interest[batch_index:(batch_index + batch_size)]
        batch = np.reshape(batch, [len(batch), image_size, image_size, 1])
        # save_images(batch, [8,8], FLAGS.save_result_path + 'batch_' + "%03d" % step + '.png')

        model.train_model(batch, step, idx, epoch_id, 100, saver, sess, writer)
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
      default=64,
      help='Size of batch'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)