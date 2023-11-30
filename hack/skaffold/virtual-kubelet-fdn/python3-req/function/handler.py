import os
import pickle
import urllib3
import json 

import tensorflow as tf
import numpy as np

from minio import Minio

from official.nlp.keras_nlp import layers as nlp_layers
from official.nlp.modeling import layers as model_layers
from official.modeling.optimization.lr_schedule import LinearWarmup

from .backend.earlyexit import EarlyExit
from .backend.model import DistributedModel

from .utils.reqmodel import RequestModel

BACKEND = ""

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  BACKEND = "EDGE"
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024+512)])
    logical_gpus = tf.config.list_logical_devices('GPU')
  except RuntimeError as e:
    print(e)
else:
  BACKEND = "CLOUD"

CUSTOM_OBJECTS = {
  "EarlyExit": EarlyExit,
  "DistributedModel": DistributedModel,
  "SelfAttentionMask": nlp_layers.SelfAttentionMask,
  "OnDeviceEmbedding": nlp_layers.OnDeviceEmbedding,
  "TransformerEncoderBlock": nlp_layers.TransformerEncoderBlock,
  "PositionEmbedding": nlp_layers.PositionEmbedding,
  "BertTokenizer": model_layers.BertTokenizer,
  "BertPackInputs": model_layers.BertPackInputs,
  "ClassificationHead": model_layers.ClassificationHead,
  "TruncatedNormal": tf.keras.initializers.TruncatedNormal,
  "EinsumDense": tf.keras.layers.experimental.EinsumDense,
  "<lambda>": lambda x: tf.keras.activations.gelu(x, approximate=True),
  "LinearWarmup": LinearWarmup,
}

def handle_fullmodel(params: RequestModel):
  """
    Loads the complete models from Minio and performs inferencing on it

    This method assumes the model is not split but rather the edge system is overloaded
    thus the cloud system has to perform the full inferencing.
  """
  client = Minio(params.miniohost, params.minioaccess, params.miniopass, secure=False, cert_check=False)
  if not client.bucket_exists(params.inputbucket):
    return {"message": "Input bucket does not exist"}
  
  model_name = f"{params.modelname}-{params.dataset}-{'ee' if params.earlyexit else 'full'}.h5"
  
  if not os.path.isfile(model_name):
    client.fget_object(params.modelstore, model_name, model_name)
  
  client.fget_object(params.inputbucket, params.inputname, params.inputname)
  model: DistributedModel = tf.keras.models.load_model(model_name, custom_objects=CUSTOM_OBJECTS, compile=False)
  with open(params.inputname, "rb") as f:
    x_in = pickle.load(f)
  model.thr = params.threshold
  output = model.predict(x_in, batch_size=params.batchsize, eager_predict=True, enable_earlyexit=params.earlyexit, use_wrapper=False)
  if model.exit_activated:
    os.remove(model_name)
    os.remove(params.inputname)
    return {"output": output.tolist(), "layer_exit": model.layer_exit, "exec_time": model._internal_timer.execution, "backend": BACKEND}
  else:
    os.remove(model_name)
    os.remove(params.inputname)
    return {"output": [out.numpy().tolist() for out in output], "exec_time": model._internal_timer.execution, "backend": BACKEND}

def handle_head(params: RequestModel):
  """Loads a model from Minio and performs inferencing on it"""
  if not gpus or params.forcefull:
    return handle_fullmodel(params)
  
  client = Minio(params.miniohost, params.minioaccess, params.miniopass, secure=False, cert_check=False)
  if not client.bucket_exists(params.inputbucket):
    return {"message": "Input bucket does not exist"}
  
  client.fget_object(params.modelstore, f"{params.modelname}-{params.dataset}-head.h5", "model.h5")
  client.fget_object(params.inputbucket, params.inputname, "input.pkl")
  model: DistributedModel = tf.keras.models.load_model("model.h5", custom_objects=CUSTOM_OBJECTS)
  with open("input.pkl", "rb") as f:
    x_in = pickle.load(f)
  
  model.thr = params.threshold
  outputs = model.predict(x_in, batch_size=params.batchsize, eager_predict=True, enable_earlyexit=params.earlyexit, use_wrapper=False)
  if model.exit_activated:
    os.remove("model.h5")
    os.remove("input.pkl")
    return {"output": outputs.tolist(), "layer_exit": model.layer_exit, "exec_time": model._internal_timer.execution, "backend": BACKEND}
  else:
    params.outputnames = model.output_names
    params.output = [output.numpy().tolist() for output in outputs]
    timeout = urllib3.Timeout(connect=180, read=180)
    json_body = json.dumps(params.dict()).encode("utf-8")
    print(json_body)
    response = urllib3.PoolManager(timeout=timeout).request("POST", params.tailendpoint, body=json_body, headers={"Content-Type": "application/json"})
    if response.status != 200:
      return {"message": "Error while requesting cloud resources"}
    return json.loads(response.data.decode("utf-8"))

def handle_tail(params: RequestModel):
  """Loads a model from Minio and performs the rest of inferencing on it"""
  client = Minio(params.miniohost, params.minioaccess, params.miniopass, secure=False, cert_check=False)
  if not client.bucket_exists(params.inputbucket):
    return {"message": "Input bucket does not exist"}
  
  client.fget_object(params.modelstore, f"{params.modelname}-{params.dataset}-tail.h5", "model.h5")
  
  model: DistributedModel = tf.keras.models.load_model("model.h5", custom_objects=CUSTOM_OBJECTS)
  
  inputs = []
  for x_in in model.input_names:
    if x_in in params.outputnames:
      idx = params.outputnames.index(x_in)
      inputs.append(np.array(params.output[params.output[idx]]))
  
  model.thr = params.threshold
  output = model.predict(inputs, batch_size=params.batchsize, eager_predict=True, enable_earlyexit=params.earlyexit, use_wrapper=False)
  os.remove("model.h5")
  os.remove("input.pkl")
  return {"output": [out.numpy().tolist() for out in output], "layer_exit": model.layer_exit, "exec_time": model._internal_timer.execution, "backend": BACKEND}

  

