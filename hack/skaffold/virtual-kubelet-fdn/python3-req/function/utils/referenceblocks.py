from keras import layers

def resnet_residual_block(x, filters, kernel_size: int = 3,
                          stride: int = 1, shortcut: bool = True,
                          axis: int = 3, epsilon: float = 1e-5,
                          activation: str = "relu"):
  """ Residual block for ResNet models """
  if shortcut:
    x_short = layers.Conv2D(4*filters, 1, strides = stride)(x)
    x_short = layers.BatchNormalization(axis = axis, epsilon = epsilon)(x_short)
  else:
    x_short = x
  x = layers.Conv2D(filters, 1, strides = stride)(x)
  x = layers.BatchNormalization(axis=axis, epsilon=epsilon)(x)
  x = layers.Activation(activation)(x)

  x = layers.Conv2D(filters, kernel_size, padding="SAME")(x)
  x = layers.BatchNormalization(axis=axis, epsilon=epsilon)(x)
  x = layers.Activation(activation)(x)

  x = layers.Conv2D(4*filters, 1)(x)
  x = layers.BatchNormalization(axis=axis, epsilon=epsilon)(x)
  x = layers.Add()([x_short, x])
  x = layers.Activation(activation)(x)
  return x
