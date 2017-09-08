import tensorflow as tf 
from layers import *

class Nets(object):   
    def __init__(self, image_size, batch_size, keep_prob):
        self.batch_size = batch_size
        self.image_size = image_size
        self.KEEP_PROB = keep_prob

    def create_generator_DCGAN(self, z, train = True):
        final_dim = 64
        start_size = 8
        filter_size = 5
        output_channel = 1

        fc1 = fc(z, z.shape[1], start_size*start_size*final_dim*8, 'fc1')
        fc1 = tf.nn.relu(batch_norm(fc1, 'd_bn1', train = train))
        fc1_reshape = tf.reshape(fc1, [-1, start_size, start_size, final_dim*8])

        dconv2 = dconv(fc1_reshape, filter_size, filter_size, 'dconv2', 
            output_shape = [self.batch_size, start_size*2, start_size*2, final_dim*4])
        bn_dconv2 = tf.nn.relu(batch_norm(dconv2, 'd_bn2', train = train))

        dconv3 = dconv(bn_dconv2, filter_size, filter_size, 'dconv3', 
            output_shape = [self.batch_size, start_size*4, start_size*4, final_dim*2])
        bn_dconv3 = tf.nn.relu(batch_norm(dconv3, 'd_bn3', train = train))

        dconv4 = dconv(bn_dconv3, filter_size, filter_size, 'dconv4', 
            output_shape = [self.batch_size, start_size*8, start_size*8, final_dim])
        bn_dconv4 = tf.nn.relu(batch_norm(dconv4, 'd_bn4', train = train))

        dconv5 = dconv(bn_dconv4, filter_size, filter_size, 'dconv5', 
            output_shape = [self.batch_size, 128, 128, output_channel])
            # output_shape = [self.batch_size, start_size*16, start_size*16, output_channel])
        generation = tf.nn.tanh(dconv5, 'd_bn5')
        return generation

    def create_discriminator_DCGAN(self, input_im):
        filter_size = 5
        start_depth = 64

        conv1 = conv(input_im, filter_size, filter_size, start_depth, 'conv1', stride_x = 2, stride_y = 2, relu = False)
        bn_conv1 = leaky_relu((batch_norm(conv1, 'c_bn1')))

        conv2 = conv(bn_conv1, filter_size, filter_size, start_depth*2, 'conv2', stride_x = 2, stride_y = 2, relu = False)
        bn_conv2 = leaky_relu((batch_norm(conv2, 'c_bn2')))

        conv3 = conv(bn_conv2, filter_size, filter_size, start_depth*4, 'conv3', stride_x = 2, stride_y = 2, relu = False)
        bn_conv3 = leaky_relu((batch_norm(conv3, 'c_bn3')))

        conv4 = conv(bn_conv3, filter_size, filter_size, start_depth*8, 'conv4', stride_x = 2, stride_y = 2, relu = False)
        bn_conv4 = leaky_relu((batch_norm(conv4, 'c_bn4')))

        bn_conv4_flatten = tf.reshape(bn_conv4, [-1, 8*8*start_depth*4])
        fc5 = fc(bn_conv4_flatten, 0, 1, 'fc5', relu = False)

        return fc5

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