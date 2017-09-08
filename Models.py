import numpy as np
import argparse
import sys, os
import scipy.io

import tensorflow as tf 

from layers import *
from GANs import *
from common import *

class GAN(object):
    def __init__(self, len_random_vector = 32, image_size = 28, batch_size = 32, save_model_path = '', save_result_path = ''):
        """
        Inputs:
        - x: tf.placeholder, for the input images
        - keep_prob: tf.placeholder, for the dropout rate
        - weights_path: path string, path to the pretrained weights,
                    (if bvlc_alexnet.npy is not in the same folder)
        """
        # self.X = x
        # self.Z = z
        self.image_size = image_size
        self.len_random_vector = len_random_vector
        self.X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1])
        self.Z = tf.placeholder(tf.float32, [None, self.len_random_vector])
        self.KEEP_PROB = tf.placeholder(tf.float32)

        self.save_model_path = save_model_path
        self.save_result_path = save_result_path

        self.batch_size = batch_size

        GAN_model = Nets(self.image_size, self.batch_size, self.KEEP_PROB)
        
        with tf.variable_scope('generator') as scope:
            self.generation = GAN_model.create_generator_DCGAN(self.Z)
            scope.reuse_variables()
            self.sample = GAN_model.create_generator_DCGAN(self.Z, train = False)
            
        with tf.variable_scope('discriminator') as scope:
            self.discrim_real = GAN_model.create_discriminator_DCGAN(self.X)
            scope.reuse_variables()
            self.disrim_gen = GAN_model.create_discriminator_DCGAN(self.generation)
            
        with tf.name_scope('loss'):
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits = self.discrim_real, labels = tf.ones_like(self.discrim_real)))
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits = self.disrim_gen, labels = tf.zeros_like(self.disrim_gen)))
            self.d_loss = d_loss_real + d_loss_fake

            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits = self.disrim_gen, labels = tf.ones_like(self.disrim_gen)))
        tf.summary.scalar('d_loss', self.d_loss)
        tf.summary.scalar('g_loss', self.g_loss)
        # with tf.name_scope('accuracy'):
        #     d_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.discrim_real), tf.round(y_)), tf.float32))

        with tf.name_scope('train'):
            d_training_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator/')]
            self.d_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1=0.5).minimize(self.d_loss, var_list = d_training_vars)

            g_training_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator/')]
            self.g_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1=0.5).minimize(self.g_loss, var_list = g_training_vars)  

    def train_model(self, batch, step, save_step, saver, session, writer):
        batch_size = self.batch_size
        len_rand_vec = self.len_random_vector
        
        _, discriminator_loss = session.run([self.d_optimizer, self.d_loss],
         feed_dict = {self.X: batch, self.Z: np.random.normal(size = (batch_size, len_rand_vec)), self.KEEP_PROB: 0.5})
        _, generator_loss = session.run([self.g_optimizer, self.g_loss],
         feed_dict = {self.Z: np.random.normal(size = (batch_size, len_rand_vec)), self.KEEP_PROB: 1.0})

        if step % save_step == 0:
          # s = sess.run(merged_summary, feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
          # writer.add_summary(s, all_data.train.epochs_completed*100 + i)
          print("Step {} Eval: {} {}".format(step, discriminator_loss, generator_loss))
          result = session.run(self.sample, {self.Z: np.random.normal(size = (batch_size, len_rand_vec))})
          save_images(result, [6,6], self.save_result_path + 'test_FCN_result_' + "%03d" % step + '.png')
          # scipy.io.savemat(self.save_result_path + 'test_FCN_result_' + "%03d" % step + '.mat', mdict = {'resultList': np.squeeze(result)})
          # saver.save(session, self.save_model_path + 'my-model', global_step = step)



        




