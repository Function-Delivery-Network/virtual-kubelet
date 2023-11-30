""" Helper functions to process datasets for the models """
import tensorflow as tf
import numpy as np

from keras.datasets import cifar10, mnist, cifar100

def process_images_cifar100():
  """ Preprocessing for the CIFAR-100 dataset """
  (img_train, label_train), (img_test, label_test) = cifar100.load_data()

  img_train = tf.keras.applications.imagenet_utils.preprocess_input(img_train, mode="tf")
  img_test = tf.keras.applications.imagenet_utils.preprocess_input(img_test, mode="tf")

  label_train = tf.keras.utils.to_categorical(label_train, 100)
  label_test = tf.keras.utils.to_categorical(label_test, 100)

  return img_train, label_train, img_test, label_test

def process_images_cifar10():
  """ Preprocessing for the CIFAR-10 dataset """
  (img_train, label_train), (img_test, label_test) = cifar10.load_data()
  #  img_train = img_train.astype(np.float32) / 255
  #  img_test = img_test.astype(np.float32) / 255

  img_train = tf.keras.applications.imagenet_utils.preprocess_input(img_train, mode="tf")
  img_test = tf.keras.applications.imagenet_utils.preprocess_input(img_test, mode="tf")
  # img_train = tf.keras.applications.resnet50.preprocess_input(img_train)
  # img_test = tf.keras.applications.resnet50.preprocess_input(img_test)

  label_train = tf.keras.utils.to_categorical(label_train, 10)
  label_test = tf.keras.utils.to_categorical(label_test, 10)
  return img_train, label_train, img_test, label_test

def process_images_mnist():
  """ Preprocessing for the MNIST dataset """
  (img_train, label_train), (img_test, label_test) = mnist.load_data() # Loading the MNIST dataset
  img_train = np.expand_dims(img_train, axis=-1) # Expanding the dimensions of the training images
  img_train = np.repeat(img_train, 3, axis=-1) # Repeating the images 3 times to get 3 channels
  img_train = img_train.astype(np.float32) /255 # Converting the images to float32 and normalizing them
  img_train = tf.image.resize(img_train, [32,32]) # Resizing the images to 32x32
  label_train = tf.keras.utils.to_categorical(label_train, 10) # Converting the labels to one-hot vectors

  # Repeating for the test dataset
  img_test = np.expand_dims(img_test, axis=-1) # Expanding the dimensions of the training images
  img_test = np.repeat(img_test, 3, axis=-1) # Repeating the images 3 times to get 3 channels
  img_test = img_test.astype(np.float32) /255 # Converting the images to float32 and normalizing them
  img_test = tf.image.resize(img_test, [32,32]) # Resizing the images to 32x32
  label_test = tf.keras.utils.to_categorical(label_test, 10) # Converting the labels to one-hot vectors
  return img_train, label_train, img_test, label_test
