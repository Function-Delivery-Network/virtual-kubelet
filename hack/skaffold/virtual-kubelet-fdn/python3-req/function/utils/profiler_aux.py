""" Defines aux profiler functions """
import tensorflow as tf
import numpy as np
import keras.layers as kr


from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.python.profiler.model_analyzer import profile

def empty_flops_counter(layer):
    return 0
def upsample_flops_counter(layer, batch_size=1):
  input_shape = (batch_size,) + layer.input_shape[1:]
  output_shape = (batch_size,) + layer.output_shape[1:]
  return np.prod(output_shape)
def relu_flops_counter(layer, batch_size=1):
  output_shape = (batch_size,) + layer.output_shape[1:]
  return np.prod(output_shape)
def linear_flops_counter(layer, batch_size=1):
  input_shape = layer.input_shape[1:]
  output_shape = layer.output_shape[1:]
  bias_flops = layer.use_bias if hasattr(layer, 'bias') else False
  bias_flops = output_shape[-1] if bias_flops else 0
  return 2*(output_shape[0]*input_shape[0]) + bias_flops
def pool_flops_counter(layer, batch_size=1):
  i_shape = layer.input_shape[1:]
  strides = layer.strides
  ks = layer.pool_size
  flops = ((i_shape[0] / strides[0]) * (i_shape[1] / strides[1]) * (ks[0] * ks[1] * i_shape[2]))
  return flops
def global_pool_flops_counter(layer, batch_size=1):
  i_shape = layer.input_shape[1:]
  flops = (i_shape[0] * i_shape[1] * i_shape[2])
  return flops
def bn_flops_counter(layer, batch_size=1):
  i_shape = layer.input_shape[1:]
  bflops = 1
  for i in range(len(i_shape)):
    bflops *= i_shape[i] 
  return bflops
def conv_flops_counter(layer, batch_size=1):
  strides = layer.strides
  ks = layer.kernel_size
  filters = layer.filters
  i_shape = layer.input_shape[1:]
  o_shape = layer.output_shape[1:]
  flops = 2 * ((filters * ks[0] * ks[1] * i_shape[2]) * (
                  (i_shape[0] / strides[0]) * (i_shape[1] / strides[1])))
  active_elem = np.prod(o_shape)
  bias_flop = o_shape[-1] * active_elem if layer.use_bias else 0
  return flops + bias_flop
def add_flops_counter(layer, batch_size=1):
  input_shape = tuple(filter(lambda x: x is not None, layer.input_shape[0]))+ (len(layer.input_shape),)
  return np.prod(input_shape)
def activation_flops_counter(layer, batch_size=1):
  input_shape = layer.input_shape[1:]
  bflops = 1
  for i in range(len(input_shape)):
    bflops *= input_shape[i] 
  return bflops
def layer_mem_req_calc(layer):
  """Returns the memory requirement of a layer.
    Reference implementation is taken from: 
    https://github.com/Mr-TalhaIlyas/Tensorflow-Keras-Model-Profiler/blob/d42bdb33feea1ce894573ca7e3a0507e7a6a1339/model_profiler/utils.py#L64
  """
  single_layer_mem = tf.as_dtype(layer.dtype).size
  layer_mem =  single_layer_mem * np.prod(layer.output_shape[1:])
  trainable_count = sum([tf.keras.backend.count_params(p) for p in layer.trainable_weights])
  non_trainable_count = sum([tf.keras.backend.count_params(p) for p in layer.non_trainable_weights])
  return layer_mem + (trainable_count + non_trainable_count) * single_layer_mem
  return layer.count_params()*4
def layer_profiler_aux(layer, batch_size: int = 1):
  """ Calculates layer-related metrics """
  layer_flop = 0
  if isinstance(layer, (kr.Conv1D, kr.Conv2D, kr.Conv3D, \
                        kr.Conv1DTranspose, kr.Conv2DTranspose, kr.Conv3DTranspose)):
    return conv_flops_counter(layer, batch_size)
  elif isinstance(layer, (kr.ReLU, kr.PReLU, kr.ELU, kr.LeakyReLU)):
    return relu_flops_counter(layer, batch_size)
  elif isinstance(layer, (kr.AveragePooling1D, kr.AveragePooling2D, kr.AveragePooling3D, \
                          kr.MaxPooling1D, kr.MaxPooling2D, kr.MaxPooling3D)):
    return pool_flops_counter(layer, batch_size)
  elif isinstance(layer, (kr.GlobalAveragePooling1D, kr.GlobalAveragePooling2D, kr.GlobalAveragePooling3D, \
                          kr.GlobalMaxPooling1D, kr.GlobalMaxPooling2D, kr.GlobalMaxPooling3D,)):
    return global_pool_flops_counter(layer, batch_size)
  elif isinstance(layer, (kr.BatchNormalization)):
    return bn_flops_counter(layer, batch_size)
  elif isinstance(layer, (kr.Dense)):
    return linear_flops_counter(layer, batch_size)
  elif isinstance(layer, (kr.UpSampling1D, kr.UpSampling2D, kr.UpSampling3D)):
    return upsample_flops_counter(layer, batch_size)
  elif isinstance(layer, kr.Add):
    return add_flops_counter(layer, batch_size)
  elif isinstance(layer, kr.Activation):
    return activation_flops_counter(layer, batch_size)
  return layer_flop
def layer_profiler(layer, eager_exec: bool = False):
  """ Calculates layer-related metrics """
  return layer_mem_req_calc(layer)
  if eager_exec:
    layer_spec = tf.function(layer.call)
    input_shape = layer.input_shape
    in_shape_spec = tf.TensorSpec(shape=(1,) + input_shape[1:]) if isinstance(input_shape, tuple) \
                    else [tf.TensorSpec(shape=(1,) + in_shape[1:]) for in_shape in input_shape]
    layer_spec = layer_spec.get_concrete_function(in_shape_spec)
    frozen_layer = convert_variables_to_constants_v2(layer_spec, lower_control_flow=False)
    opts = ProfileOptionBuilder(ProfileOptionBuilder.float_operation()).with_empty_output().build()
    graph_info = profile(frozen_layer.graph, options=opts)
    flops = graph_info.total_float_ops if graph_info.total_float_ops > 0\
            else layer_profiler_aux(layer)
  else:
    flops = layer_profiler_aux(layer)
  data_exchange = 0
  if isinstance(layer.output_shape, list):
    for output in layer.output_shape:
      data_exchange += np.prod((1,) + tuple(filter(lambda x: x is not None, output)))
  elif isinstance(layer.output_shape, dict):
    for output in layer.output_shape.values():
      if isinstance(output, tuple):
        shapes = tuple(filter(lambda x: x is not None, output))
        data_exchange += np.prod((1,) + shapes) if len(shapes) > 1 else shapes[0]
      elif isinstance(output, list):
        for out in output:
          if isinstance(out, tuple):
            shapes = tuple(filter(lambda x: x is not None, out))
          else:
            shapes = out
          data_exchange += np.prod((1,) + shapes) if len(shapes) > 1 else shapes[0]      
  else:
    shapes = tuple(filter(lambda x: x is not None, layer.output_shape))
    data_exchange = np.prod((1,) + shapes) if len(shapes) > 1 else (shapes[0] if len(shapes) > 0 else 0)
  return flops, data_exchange, layer_mem_req_calc(layer)