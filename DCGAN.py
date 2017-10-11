# File: DCGAN.py
# Author: Qian Ge <geqian1001@gmail.com>

import argparse

import numpy as np
import tensorflow as tf

import tensorcv
from tensorcv.dataflow import *
from tensorcv.callbacks import *
from tensorcv.predicts import *
from tensorcv.predicts.simple import SimpleFeedPredictor
from tensorcv.train.config import GANTrainConfig
from tensorcv.train.simple import GANFeedTrainer
from tensorcv.algorithms.GAN.DCGAN import Model

import config

def get_config(FLAGS):
    if FLAGS.mnist:
        dataset_train = MNIST('train', data_dir=config.data_dir, normalize='tanh')
    elif FLAGS.cifar:
        dataset_train = CIFAR(data_dir=config.data_dir, normalize='tanh')
    elif FLAGS.matlab:
        mat_name_list = FLAGS.mat_name
        dataset_train = MatlabData(
                               mat_name_list=mat_name_list,
                               data_dir=config.data_dir,
                               normalize='tanh')
    elif FLAGS.image:
        # dataset_train = ImageData('.png', data_dir = config.data_dir,
        #                            normalize = 'tanh')
        dataset_train = ImageFromFile(FLAGS.type, 
                                data_dir=config.data_dir, 
                                normalize='tanh')

    inference_list = InferImages('generate_image', prefix='gen')
    random_feed = RandomVec(len_vec=FLAGS.len_vec)
    
    return GANTrainConfig(
            dataflow = dataset_train, 
            model = Model(input_vec_length=FLAGS.len_vec, 
                          learning_rate=[0.0002, 0.0002]),
            monitors = TFSummaryWriter(),
            discriminator_callbacks=[
                ModelSaver(periodic=100), 
                CheckScalar(['d_loss_check', 'g_loss_check'], periodic=10),
              ],
            generator_callbacks = [
                        GANInference(inputs=random_feed, periodic=100, 
                                    inferencers=inference_list),
                    ],              
            batch_size=FLAGS.batch_size, 
            max_epoch=100,
            summary_d_periodic=10, 
            summary_g_periodic=10,
            default_dirs=config)

def get_predictConfig(FLAGS):
    random_feed = RandomVec(len_vec=FLAGS.len_vec)
    prediction_list = PredictionImage('generate_image', 
                                      'test', merge_im=True, tanh=True)
    im_size = [FLAGS.h, FLAGS.w]
    return PridectConfig(
                         dataflow=random_feed,
                         model = Model(input_vec_length=FLAGS.len_vec, 
                                       num_channels=FLAGS.input_channel, 
                                       im_size=im_size),
                         model_name='model-100', 
                         predictions=prediction_list,
                         batch_size=FLAGS.batch_size,
                         default_dirs=config)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--len_vec', default=100, type=int,
                        help='Length of input random vector')
    parser.add_argument('--input_channel', default=1, type=int,
                        help='Number of image channels')
    parser.add_argument('--h', default=32, type=int,
                        help='Heigh of input images')
    parser.add_argument('--w', default=32, type=int,
                        help='Width of input images')
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--predict', action='store_true', 
                        help = 'Run prediction')
    parser.add_argument('--train', action='store_true', 
                        help='Train the model')

    parser.add_argument('--mnist', action='store_true',
                        help='Run on MNIST dataset')
    parser.add_argument('--cifar', action='store_true',
                        help='Run on CIFAR dataset')

    parser.add_argument('--matlab', action='store_true',
                        help='Run on dataset of .mat files')
    parser.add_argument('--mat_name', type=str, default = None,
                        help = 'Name of mat to be loaded from .mat file')

    parser.add_argument('--image', action='store_true',
                        help='Run on dataset of image files')
    parser.add_argument('--type', type=str,
                        help='Image file type')

    return parser.parse_args()

if __name__ == '__main__':

    FLAGS = get_args()

    if FLAGS.train:
        config = get_config(FLAGS)
        GANFeedTrainer(config).train()
    elif FLAGS.predict:
        config = get_predictConfig(FLAGS)
        SimpleFeedPredictor(config).run_predict()



 