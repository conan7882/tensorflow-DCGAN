## Deep Convolutional Generative Adversarial Networks (DCGAN)


TensorFlow implementation of [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434). 

The DCGAN model is defined [here](https://github.com/conan7882/DeepVision-tensorflow/tree/master/algorithms/GAN).

Please refer to the docs (coming soon) for custom configuration of the model and the callbacks setup.



## Requirements
- Python 3.3+
- [Tensorflow 1.0+](https://www.tensorflow.org/)
- [TensorCV](https://github.com/conan7882/DeepVision-tensorflow)


## Usage
### Config path
All directories are setup in *config.py*.

Before training, put all the training images in *`config.data_dir`*.

**Please note, all the images should have the same size.**


### Run script

You can run this script on CIFAR10, MNIST dataset as well as your own dataset in format of Matlab .mat files and image files.

To train a model on CIFAR10 and MNIST dataset:

	$ python DCGAN.py --train --cifar --batch_size 32
	$ python DCGAN.py --train --mnist --batch_size 32


To train on your own dataset:

.mat files:

	$ python DCGAN.py --train --matlab --batch_size 32 --mat_name MAT_NAME_IN_MAT_FILE

images files:

	$ python DCGAN.py --train --image --batch_size 32 --type IMAGE_FILE_EXTENSION(start with '.')
	 
To test using an existing model, size and channels of images used for training the model need to be specified, and the batch size has to be the same as training:

	$ python DCGAN.py --predict --batch_size SAME_AS_TRAINING --h IMAGE_HEIGHT\
	 --w IMAGE_WIDTH --input_channel NUM_INPUT_CHANNEL
	


## Training Details
- Both discriminator and generator use learning rate 0.0002
- To avoid small discriminator loss, update generator twice for each update of discriminator, as suggested [here](https://github.com/carpedm20/DCGAN-tensorflow).
- init

## Default Summary
### Scalar:
- loss of generator and discriminator

### Histogram:
- gradients of generator and discriminator
- discriminator output for real image and generated image

### Image
- real image and generated image

## Costum Configuration
*details can be found in docs (comming soon)*
### Available callbacks:

- TrainSummary()
- CheckScalar()
- GANInference()
 
### Available inferencer:
- InferImages()

## Results

### CIFAR10
![cifar_result1](fig/cifar_result.png)

### MNIST

![MNIST_result1](fig/mnist_result.png)

*More results will be added later.*

## Training Details

## Author
Qian Ge





