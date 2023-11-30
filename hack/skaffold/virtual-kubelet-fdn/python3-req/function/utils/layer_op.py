""" Auxiliary layer operations to comply with size """
from keras import backend

def _make_divisible(v, divisor, min_value=None):
  """ Make channels divisible by divisor"""
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v

def correct_pad(inputs, kernel_size):
  """Returns a tuple for zero-padding for 2D convolution with downsampling.

  Args:
    inputs: Input tensor.
    kernel_size: An integer or tuple/list of 2 integers.

  Returns:
    A tuple.
  """
  img_dim = 2 if backend.image_data_format() == "channels_first" else 1
  input_size = backend.int_shape(inputs)[img_dim : (img_dim + 2)]
  if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)
  if input_size[0] is None:
      adjust = (1, 1)
  else:
      adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
  correct = (kernel_size[0] // 2, kernel_size[1] // 2)
  return (
      (correct[0] - adjust[0], correct[0]),
      (correct[1] - adjust[1], correct[1]),
  )