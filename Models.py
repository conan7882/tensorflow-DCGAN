import numpy as np
import argparse
import sys, os
import scipy.io
from scipy import interpolate

import tensorflow as tf 

from layers import *
from GANs import *
from common import *

class GAN(object):
    def __init__(self, len_random_vector = 32, image_size = 28, batch_size = 32, num_channel = 1,
        d_learning_rate = 0.0002, g_learning_rate = 0.0002,
        save_model_path = '', save_result_path = '', data_type = 'default',
        flag_debug = True):
        """
        Inputs:
        - x: tf.placeholder, for the input images
        - keep_prob: tf.placeholder, for the dropout rate
        - weights_path: path string, path to the pretrained weights,
                    (if bvlc_alexnet.npy is not in the same folder)
        """
        self.flag_debug = flag_debug

        self.image_size = image_size
        self.len_random_vector = len_random_vector
        self.X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, num_channel])
        self.Z = tf.placeholder(tf.float32, [None, self.len_random_vector])
        self.KEEP_PROB = tf.placeholder(tf.float32)

        self.save_model_path = save_model_path
        self.save_result_path = save_result_path

        self.batch_size = batch_size

        GAN_model = Nets(self.image_size, self.batch_size, self.KEEP_PROB, data_type = data_type, num_channel = num_channel)
        
        # with tf.variable_scope('generator') as scope:
        self.generation = GAN_model.create_generator_DCGAN(self.Z)
        # scope.reuse_variables()
        self.sample = GAN_model.create_generator_DCGAN(self.Z, train = False, reuse = True)
            
        self.discrim_real = GAN_model.create_discriminator_DCGAN(self.X)
        self.disrim_gen = GAN_model.create_discriminator_DCGAN(self.generation, reuse = True)
        
        d_real_summary = tf.summary.histogram("d_", tf.nn.sigmoid(self.discrim_real))
        d_fake_summary = tf.summary.histogram("d", tf.nn.sigmoid(self.disrim_gen))
        G_summary = tf.summary.image("G", self.generation)
            
        with tf.name_scope('loss'):
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits = self.discrim_real, labels = tf.ones_like(self.discrim_real)))
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits = self.disrim_gen, labels = tf.zeros_like(self.disrim_gen)))
            self.d_loss = d_loss_real + d_loss_fake

            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits = self.disrim_gen, labels = tf.ones_like(self.disrim_gen)))
        d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
        g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
        # with tf.name_scope('accuracy'):
        #     d_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.discrim_real), tf.round(y_)), tf.float32))

        with tf.name_scope('train'):
            d_training_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator/')]
            d_optimizer = tf.train.AdamOptimizer(learning_rate = d_learning_rate, beta1=0.5)
            d_grads = d_optimizer.compute_gradients(self.d_loss, var_list = d_training_vars)
            self.d_optimizer = d_optimizer.apply_gradients(d_grads)
            # self.d_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1=0.5).minimize(self.d_loss, var_list = d_training_vars)

            g_training_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator/')]
            g_optimizer = tf.train.AdamOptimizer(learning_rate = g_learning_rate, beta1=0.5)
            g_grads = g_optimizer.compute_gradients(self.g_loss, var_list = g_training_vars)
            self.g_optimizer = g_optimizer.apply_gradients(g_grads)
            # self.g_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1=0.5).minimize(self.g_loss, var_list = g_training_vars)  

        d_var_summary = [tf.summary.histogram(var.name, var) for var in d_training_vars]
        g_var_summary = [tf.summary.histogram(var.name, var) for var in g_training_vars]
        d_grads_summary = [tf.summary.histogram('gradient/' + var.name, grad) for grad, var in d_grads]
        g_grads_summary = [tf.summary.histogram('gradient/' + var.name, grad) for grad, var in g_grads]

        self.g_sum = tf.summary.merge([g_loss_summary, G_summary, g_grads_summary])
        self.d_sum = tf.summary.merge([d_loss_summary, d_real_summary, d_fake_summary, d_grads_summary])

    def test_model(self, session):
        batch_size = self.batch_size
        len_rand_vec = self.len_random_vector
        test_vec_1 = np.random.normal(size = (batch_size, len_rand_vec))
        test_vec_2 = np.random.normal(size = (batch_size, len_rand_vec))

        result_1 = session.run(self.sample, {self.Z: test_vec_1})
        result_2 = session.run(self.sample, {self.Z: test_vec_2})
        if batch_size == 32:
            sample_image_grid_size = [6,6]
        else:
            sample_image_grid_size = [8,8]
        save_images(result_1, sample_image_grid_size, self.save_result_path + 'test_result_01.png')
        save_images(result_2, sample_image_grid_size, self.save_result_path + 'test_result_02.png')

        x = [0, batch_size-1]
        y = np.array([test_vec_1[0], test_vec_2[0]])
        f = interpolate.interp1d(x, y, axis = 0)
        x_interp = range(0, batch_size)
        y_interp = f(x_interp)
        result_interp = session.run(self.sample, {self.Z: y_interp})
        save_images(result_interp, sample_image_grid_size, self.save_result_path + 'test_result_interp.png')

    def train_model(self, batch, step, idx, epoch_id, save_step, saver, session, writer):
        batch_size = self.batch_size
        len_rand_vec = self.len_random_vector
        
        for i in range(0,1):
            if self.flag_debug:
                _, discriminator_loss, d_sum = session.run([self.d_optimizer, self.d_loss, self.d_sum],
                    feed_dict = {self.X: batch, self.Z: np.random.normal(size = (batch_size, len_rand_vec)), self.KEEP_PROB: 0.5})
                writer.add_summary(d_sum, step)
            else:
                _, discriminator_loss = session.run([self.d_optimizer, self.d_loss],
                    feed_dict = {self.X: batch, self.Z: np.random.normal(size = (batch_size, len_rand_vec)), self.KEEP_PROB: 0.5})

        for i in range(0,2):
            if self.flag_debug:
                _, generator_loss, g_sum = session.run([self.g_optimizer, self.g_loss, self.g_sum],
                    feed_dict = {self.Z: np.random.normal(size = (batch_size, len_rand_vec)), self.KEEP_PROB: 1.0})
                writer.add_summary(g_sum, step)
            else:
                _, generator_loss = session.run([self.g_optimizer, self.g_loss],
                    feed_dict = {self.Z: np.random.normal(size = (batch_size, len_rand_vec)), self.KEEP_PROB: 1.0})


        if step % save_step == 0:
          # discriminator_loss, generator_loss, d_sum, g_sum = session.run([self.d_loss, self.g_loss, self.d_sum, self.g_sum],
          #   feed_dict = {self.X: batch, self.Z: np.random.normal(size = (batch_size, len_rand_vec)), self.KEEP_PROB: 1.0})
          # writer.add_summary(d_sum, step)
          # writer.add_summary(g_sum, step)

          print("Epoch {} Step {} Eval: {} {}".format(epoch_id, step, discriminator_loss, generator_loss))
          result = session.run(self.sample, {self.Z: np.random.normal(size = (batch_size, len_rand_vec))})
          if batch_size == 32:
            sample_image_grid_size = [6,6]
          else:
            sample_image_grid_size = [8,8]

          save_images(result, sample_image_grid_size, self.save_result_path + 'test_result_' + "%03d" % step + '.png')
          # scipy.io.savemat(self.save_result_path + 'test_FCN_result_' + "%03d" % step + '.mat', mdict = {'resultList': np.squeeze(result)})
          saver.save(session, self.save_model_path + 'my-model', global_step = step)



        




