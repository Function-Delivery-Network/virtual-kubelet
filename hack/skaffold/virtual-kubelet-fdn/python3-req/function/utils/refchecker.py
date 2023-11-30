""" Helper module for reference checking """
import keras

from keras.utils import tf_inspect

def comply_with_ref_argspec(obj, params):
  """ Generate args and kwargs for a given object """
  argspect = tf_inspect.getfullargspec(obj)
  args = list()
  for idx, arg in enumerate(argspect.args[1:]):
    if arg in params:
      args.append(params.pop(arg))
    else:
      if argspect.defaults and len(argspect.defaults) > idx:
        args.append(argspect.defaults[idx])
  return args, params

def load_ref_object(name, base_module = keras):
  """ Searches and returns the object from a given name """
  mod_dir = dir(base_module)
  for subdir in mod_dir:
    if hasattr(getattr(base_module, subdir, None), name):
      return getattr(getattr(base_module, subdir), name)
  if hasattr(base_module, name):
    return getattr(base_module, name)
  return None
