"""Defines the preprocessing for the provided datasets"""
import os

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from keras import layers
from official.nlp.modeling.layers import BertTokenizer, BertPackInputs

class BertInputProcessor(layers.Layer):
  """ Preprocesses the input for the Bert model """
  def __init__(self, tokenizer, packer):
    super().__init__()
    self.tokenizer = tokenizer
    self.packer = packer
  def call(self, inputs):
    """ Preprocesses the input for the Bert model """
    tok1 = self.tokenizer(inputs["sentence1"])
    tok2 = self.tokenizer(inputs["sentence2"])
    packed = self.packer([tok1, tok2])
    if "label" in inputs:
      return packed, inputs["label"]
    else:
      return packed

def mrpc_info(batch_size:int = 32):
  _, info = tfds.load(
    "glue/mrpc", with_info=True, batch_size=batch_size
  )
  return info

def mrpc(max_length:int =128, batch_size:int = 32):
  """ Loads the GLUE/MRPC dataset """
  gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
  tokenizer = BertTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
    lower_case=True
  )

  packer = BertPackInputs(
    seq_length=max_length,
    special_tokens_dict=tokenizer.get_special_tokens_dict(),
  )

  preprocessor = BertInputProcessor(tokenizer, packer)
  glue, info = tfds.load(
    "glue/mrpc", with_info=True, batch_size=batch_size
  )
  return glue["train"].map(preprocessor), glue["validation"].map(preprocessor)

def cifar10(*args, **kwargs):
  """ Loads the CIFAR10 dataset """
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_train = tf.keras.applications.imagenet_utils.preprocess_input(x_train)
  y_train = tf.keras.utils.to_categorical(y_train, 10)
  return x_train, y_train

def cifar100(*args, **kwargs):
  """ Loads the CIFAR100 dataset """
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

  x_train = tf.keras.applications.imagenet_utils.preprocess_input(x_train)
  y_train = tf.keras.utils.to_categorical(y_train, 100)
  return x_train, y_train