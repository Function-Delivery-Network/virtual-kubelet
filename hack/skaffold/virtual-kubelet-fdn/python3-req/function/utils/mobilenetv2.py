""" Implementation of MobileNetV2 """
from keras import layers as keras_layers
from keras import backend as keras_backend

from utils.layer_op import _make_divisible, correct_pad

from ..backend.model import DistributedModel

def inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
  channel_axis = -1
  in_channels = keras_backend.int_shape(inputs)[channel_axis]
  pointwise_conv_filters = int(filters * alpha)
  pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
  x = inputs
  if block_id:
    x = keras_layers.Conv2D(
      expansion * in_channels,
      kernel_size=1,
      padding="same",
      use_bias=False,
      activation=None,
    )(x)
    x = keras_layers.BatchNormalization(
      axis=channel_axis,
      momentum=0.999,
      epsilon=1e-3,
    )(x)
    x = keras_layers.ReLU(6.0)(x)
  
  if stride == 2:
    x = keras_layers.ZeroPadding2D(
      padding=correct_pad(x, 3),
    )(x)
  
  x = keras_layers.DepthwiseConv2D(
    kernel_size=3,
    strides=stride,
    activation=None,
    use_bias=False,
    padding="same" if stride == 1 else "valid",
  )(x)
  x = keras_layers.BatchNormalization(
    axis=channel_axis,
    momentum=0.999,
    epsilon=1e-3,
  )(x)
  x = keras_layers.ReLU(6.0)(x)
  x = keras_layers.Conv2D(
    pointwise_filters,
    kernel_size=1,
    padding="same",
    use_bias=False,
    activation=None,
  )(x)
  x = keras_layers.BatchNormalization(
    axis=channel_axis,
    momentum=0.999,
    epsilon=1e-3,
  )(x)
  if in_channels == pointwise_filters and stride == 1:
    return keras_layers.Add()([inputs, x])
  return x

def generate_mobilenetv2(
  input_shape,
  alpha=1.0,
  pooling=None,
  classes=1000,
  classifier_activation="softmax",
):
  net_input = keras_layers.Input(shape=input_shape)
  block_filters = _make_divisible(32 * alpha, 8)
  x = keras_layers.Conv2D(
        block_filters,
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
      )(net_input)
  x = keras_layers.BatchNormalization(
        axis=-1,
        momentum=0.999,
        epsilon=1e-3,
      )(x)
  x = keras_layers.ReLU(6.0)(x)
  x = inverted_res_block(
    x, filters = 16, alpha = alpha, stride = 1, expansion = 1, block_id = 0
  )

  x = inverted_res_block(
    x, filters = 24, alpha = alpha, stride = 2, expansion = 6, block_id = 1
  )
  x = inverted_res_block(
    x, filters = 24, alpha = alpha, stride = 1, expansion = 6, block_id = 2
  )

  x = inverted_res_block(
    x, filters = 32, alpha = alpha, stride = 2, expansion = 6, block_id = 3
  )
  x = inverted_res_block(
    x, filters = 32, alpha = alpha, stride = 1, expansion = 6, block_id = 4
  )
  x = inverted_res_block(
    x, filters = 32, alpha = alpha, stride = 1, expansion = 6, block_id = 5
  )

  x = inverted_res_block(
    x, filters = 64, alpha = alpha, stride = 2, expansion = 6, block_id = 6
  )
  x = inverted_res_block(
    x, filters = 64, alpha = alpha, stride = 1, expansion = 6, block_id = 7
  )
  x = inverted_res_block(
    x, filters = 64, alpha = alpha, stride = 1, expansion = 6, block_id = 8
  )
  x = inverted_res_block(
    x, filters = 64, alpha = alpha, stride = 1, expansion = 6, block_id = 9
  )

  x = inverted_res_block(
    x, filters = 96, alpha = alpha, stride = 1, expansion = 6, block_id = 10
  )
  x = inverted_res_block(
    x, filters = 96, alpha = alpha, stride = 1, expansion = 6, block_id = 11
  )
  x = inverted_res_block(
    x, filters = 96, alpha = alpha, stride = 1, expansion = 6, block_id = 12
  )

  x = inverted_res_block(
    x, filters = 160, alpha = alpha, stride = 2, expansion = 6, block_id = 13
  )
  x = inverted_res_block(
    x, filters = 160, alpha = alpha, stride = 1, expansion = 6, block_id = 14
  )
  x = inverted_res_block(
    x, filters = 160, alpha = alpha, stride = 1, expansion = 6, block_id = 15
  )

  x = inverted_res_block(
    x, filters = 320, alpha = alpha, stride = 1, expansion = 6, block_id = 16
  )

  if alpha > 1.0:
    last_block_filters = _make_divisible(1280 * alpha, 8)
  else:
    last_block_filters = 1280
  
  x = keras_layers.Conv2D(
    last_block_filters,
    kernel_size=1,
    use_bias=False,
  )(x)
  x = keras_layers.BatchNormalization(
    axis=-1,
    momentum=0.999,
    epsilon=1e-3,
  )(x)
  x = keras_layers.ReLU(6.0)(x)
  x = keras_layers.GlobalAveragePooling2D()(x)
  x = keras_layers.Dense(
    classes,
    activation=classifier_activation,
  )(x)
  return DistributedModel(inputs=net_input, outputs=x)
