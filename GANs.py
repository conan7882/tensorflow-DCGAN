import math

import tensorflow as tf 

from layers import *

class Nets(object):   
    def __init__(self, image_size, batch_size, keep_prob, num_channel = 1, data_type = 'defaul'):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channel = num_channel
        self.KEEP_PROB = keep_prob
        self.DATA_TYPE = data_type

    def create_generator_DCGAN(self, z, train = True):
        return generator_MNIST(z, self.image_size, self.image_size, self.batch_size, train = train, output_channel = self.num_channel)

    def create_discriminator_DCGAN(self, input_im):
        return discriminator_MNIST(input_im, self.batch_size)

    def create_generator_MNIST(self, z):
        down_sample_size = int(self.image_size/4)
        
        fc1 = fc(z, z.shape[1], 1024, 'fc1')
        fc2 = fc(fc1, 1024, down_sample_size*down_sample_size*64, 'fc2')
        fc2_reshape = tf.reshape(fc2, [-1, down_sample_size, down_sample_size, 64])
        dconv3 = dconv(fc2_reshape, 5, 5, 'dconv3', output_shape = [self.batch_size, down_sample_size*2, down_sample_size*2, 32])
        dconv3_relu = tf.nn.relu(dconv3)

        # shape_Z = tf.shape(self.Z)
        # dconv4_shape = tf.stack([shape_X[0], shape_X[1], shape_X[2], self.NUM_CLASSES])
        dconv4 = dconv(dconv3_relu, 5, 5, 'dconv4', output_shape = [self.batch_size, self.image_size, self.image_size, 1])
        dconv4_reshape = tf.reshape(dconv4, [-1, self.image_size, self.image_size, 1])

        self.generation = tf.sigmoid(dconv4_reshape)


    def create_discriminator_MNIST(self, input_im):
        down_sample_size = int(self.image_size/4)

        conv1 = conv(input_im, 5, 5, 32, 'conv1')
        pool1 = max_pool(conv1, name = 'pool1')

        conv2 = conv(pool1, 5, 5, 64, 'conv2')
        pool2 = max_pool(conv2, name = 'pool2')

        pool2_flat = tf.reshape(pool2, [-1, down_sample_size*down_sample_size*64])

        fc1 = fc(pool2_flat, down_sample_size*down_sample_size*64, 1024, 'fc1')
        dropout_fc1 = dropout(fc1, self.KEEP_PROB)

        fc2 = fc(dropout_fc1, 1024, 1, 'fc2', relu = False)
        # tf.sigmoid(fc2)
        return fc2

def generator_MNIST(z, out_length, out_width, batch_size, output_channel = 1, train = True):

    final_dim = 64
    filter_size = 5

    d_height_2, d_width_2 = deconv_size(out_length, out_width)
    d_height_4, d_width_4 = deconv_size(d_height_2, d_width_2)
    d_height_8, d_width_8 = deconv_size(d_height_4, d_width_4)
    d_height_16, d_width_16 = deconv_size(d_height_8, d_width_8)

    with tf.variable_scope('fc1') as scope:
        fc1 = fc(z, 0, d_height_16*d_width_16*final_dim*8, 'fc')
        fc1 = tf.nn.relu(batch_norm(fc1, 'd_bn', train = train))
        fc1_reshape = tf.reshape(fc1, [-1, d_height_16, d_width_16, final_dim*8])

    with tf.variable_scope('dconv2') as scope:
        dconv2 = dconv(fc1_reshape, filter_size, filter_size, 'dconv', 
            output_shape = [batch_size, d_height_8, d_width_8, final_dim*4])
        bn_dconv2 = tf.nn.relu(batch_norm(dconv2, 'd_bn', train = train))

    with tf.variable_scope('dconv3') as scope:
        dconv3 = dconv(bn_dconv2, filter_size, filter_size, 'dconv', 
            output_shape = [batch_size, d_height_4, d_width_4, final_dim*2])
        bn_dconv3 = tf.nn.relu(batch_norm(dconv3, 'd_bn', train = train))

    with tf.variable_scope('dconv4') as scope:
        dconv4 = dconv(bn_dconv3, filter_size, filter_size, 'dconv', 
            output_shape = [batch_size, d_height_2, d_width_2, final_dim])
        bn_dconv4 = tf.nn.relu(batch_norm(dconv4, 'd_bn', train = train))

    with tf.variable_scope('dconv5') as scope:
        dconv5 = dconv(bn_dconv4, filter_size, filter_size, 'dconv', 
            output_shape = [batch_size, out_length, out_width, output_channel])
        bn_dconv5 = batch_norm(dconv5, 'd_bn', train = train)

    generation = tf.nn.tanh(bn_dconv5, 'gen_out')
    return generation

def discriminator_MNIST(input_im, batch_size):
    filter_size = 5
    start_depth = 64

    with tf.variable_scope('conv1') as scope:
        conv1 = conv(input_im, filter_size, filter_size, start_depth, 'conv', stride_x = 2, stride_y = 2, relu = False)
        bn_conv1 = leaky_relu((batch_norm(conv1, 'c_bn')))

    with tf.variable_scope('conv2') as scope:
        conv2 = conv(bn_conv1, filter_size, filter_size, start_depth*2, 'conv', stride_x = 2, stride_y = 2, relu = False)
        bn_conv2 = leaky_relu((batch_norm(conv2, 'c_bn')))

    with tf.variable_scope('conv3') as scope:
        conv3 = conv(bn_conv2, filter_size, filter_size, start_depth*4, 'conv', stride_x = 2, stride_y = 2, relu = False)
        bn_conv3 = leaky_relu((batch_norm(conv3, 'c_bn')))

    with tf.variable_scope('conv4') as scope:
        conv4 = conv(bn_conv3, filter_size, filter_size, start_depth*8, 'conv', stride_x = 2, stride_y = 2, relu = False)
        bn_conv4 = leaky_relu((batch_norm(conv4, 'c_bn')))
        bn_conv4_shape = bn_conv4.get_shape().as_list()
        bn_conv4_flatten = tf.reshape(bn_conv4, [batch_size, bn_conv4_shape[1]*bn_conv4_shape[2]*bn_conv4_shape[3]])

    with tf.variable_scope('fc5') as scope:
        fc5 = fc(bn_conv4_flatten, 0, 1, 'fc', relu = False)

    return fc5

def deconv_size(input_height, input_width, stride = 2):
    return int(math.ceil(float(input_height) / float(stride))), int(math.ceil(float(input_height) / float(stride)))