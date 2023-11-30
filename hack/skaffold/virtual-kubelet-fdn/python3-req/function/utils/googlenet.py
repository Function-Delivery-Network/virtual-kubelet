""" Describes a GoogleNet model for image classification"""

from keras import layers

from ..backend.model import DistributedModel

def inception(
  x_in,
  filters_1x1,
  filters_3x3_reduce,
  filters_3x3,
  filters_5x5_reduce,
  filters_5x5,
  filters_pool,
):
  """ Describes the inception blocks of GoogleNet """
  path1 = layers.Conv2D(
    filters_1x1, (1, 1), padding='same', activation='relu'
  )(x_in)

  path2 = layers.Conv2D(
    filters_3x3_reduce, (1, 1), padding='same', activation='relu'
  )(x_in)
  path2 = layers.Conv2D(
    filters_3x3, (3, 3), padding='same', activation='relu'
  )(path2)

  path3 = layers.Conv2D(
    filters_5x5_reduce, (1, 1), padding='same', activation='relu'
  )(x_in)
  path3 = layers.Conv2D(
    filters_5x5, (5, 5), padding='same', activation='relu'
  )(path3)

  path4 = layers.MaxPool2D(
    (3, 3), strides=(1, 1), padding='same'
  )(x_in)

  path4 = layers.Conv2D(
    filters_pool, (1, 1), padding='same', activation='relu'
  )(path4)

  return layers.concatenate([path1, path2, path3, path4], axis=3)

def generate_googlenet(input_shape, classes = 1000, padding="same"):
  """" Generates the GoogleNet model """
  x_in = layers.Input(shape=input_shape)
  x = layers.Resizing(224, 224, interpolation="bicubic")(x_in)

  x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding=padding, activation='relu')(x)
  x = layers.MaxPool2D((3, 3), strides=(2, 2))(x)
  x = layers.Conv2D(64, (1, 1), strides=(1, 1), padding=padding, activation='relu')(x)
  x = layers.Conv2D(192, (3, 3), strides=(1, 1), padding=padding, activation='relu')(x)
  x = layers.MaxPool2D((3, 3), strides=(2, 2))(x)

  x = inception(x, 64, 96, 128, 16, 32, 32)
  x = inception(x, 128, 128, 192, 32, 96, 64)
  x = layers.MaxPool2D((3, 3), strides=(2, 2))(x)

  x = inception(x, 192, 96, 208, 16, 48, 64)

  x1 = layers.AveragePooling2D((5, 5), strides=3)(x)
  x1 = layers.Conv2D(128, (1, 1), padding=padding, activation='relu')(x1)
  x1 = layers.Flatten()(x1)
  x1 = layers.Dense(1024, activation='relu')(x1)
  x1 = layers.Dropout(0.7)(x1)
  x1 = layers.Dense(classes, activation='softmax')(x1)

  x = inception(x, 160, 112, 224, 24, 64, 64)
  x = inception(x, 128, 128, 256, 24, 64, 64)
  x = inception(x, 112, 144, 288, 32, 64, 64)

  x2 = layers.AveragePooling2D((5, 5), strides=3)(x)
  x2 = layers.Conv2D(128, (1, 1), padding=padding, activation='relu')(x2)
  x2 = layers.Flatten()(x2)
  x2 = layers.Dense(1024, activation='relu')(x2)
  x2 = layers.Dropout(0.7)(x2)
  x2 = layers.Dense(classes, activation='softmax')(x2)

  x = inception(x, 256, 160, 320, 32, 128, 128)
  x = layers.MaxPool2D((3, 3), strides=(2, 2))(x)

  x = inception(x, 256, 160, 320, 32, 128, 128)
  x = inception(x, 384, 192, 384, 48, 128, 128)

  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dropout(0.4)(x)
  x = layers.Dense(classes, activation='softmax')(x)

  return DistributedModel(inputs=x_in, outputs=[x, x1, x2])
