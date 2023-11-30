""" Defines the functional model class for Early Exit """
import os
import pickle
import time
from itertools import cycle

import tensorflow as tf
from keras import backend
from keras.engine.functional import Functional

from .earlyexit import EarlyExit
from ..utils.contexttimer import ExeTimer
from ..utils.profiler_aux import layer_profiler
from ..utils.metricscollector import installed_packages, collect_metrics
from ..utils.exceptions import EarlyExitException
from ..utils.execmodes import ExecutionMode, ProfileMode

class DistributedModel(Functional):
  """ Defines the distributed model class """
  def __init__(self, *args, exec_mode: ExecutionMode = ExecutionMode.NORMAL_EXEC, **kwargs):
    """ Initializes the distributed model class """
    super(DistributedModel, self).__init__(*args, **kwargs)
    self._enable_earlyexit = False
    self.exec_mode: ExecutionMode = exec_mode
    self.profile_mode: ProfileMode = ProfileMode.DISABLED
    self._internal_timer: ExeTimer = ExeTimer()
    self.profile_data: dict = {"layer_data": {}, "depth_data": {}}
    self.thr: float = 0.5
    self.run_until: int = -1
    self.jetson_stats = None
    self.acc_energy = 0
    self.power = 0
    self.last_time = 0
    self.offset = 0
    self.evaluator = is_high_watermark
    self.exit_activated = False
    self.layer_exit = None
  def setup_jtop(self, interval: float = 0.1):
    """ Sets up the jtop module for power profiling """
    if 'jetson-stats' in installed_packages:
      from jtop import jtop
      self.jetson_stats = jtop(interval)
    else:
      self.jetson_stats = None
    self.acc_energy = 0
    self.power = 0
  def accumulate_power(self, jetson):
    """ Callback for Jetson power consumption """
    interval = time.time()
    self.power = jetson.power['tot']['power']
    self.acc_energy += (self.power)*(interval - self.last_time)
    self.last_time = interval
  def call(self, inputs, training=None, mask=None):
    """ Defines the sub-calls depending on the execution mode """
    fn_map = {ExecutionMode.PROFILE:      self._profile_internal_graph,
              ExecutionMode.PARTIAL_RUN:  self._run_partial_graph,
              ExecutionMode.NORMAL_EXEC:  self._run_internal_graph}
    fn_call = fn_map[self.exec_mode]
    return fn_call(inputs, training, mask)
  def _toggle_internal_early_exit(self):
    """ Traverses all the layers and enables early exits"""
    for layer in self._self_tracked_trackables:
      if isinstance(layer, EarlyExit):
        layer.predict = not layer.predict
  def predict(self,
    x,
    batch_size=None,
    verbose="auto",
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    eager_predict=False,
    enable_earlyexit=True,
    use_wrapper = False,
  ):
    """ Function override for keras.layers.Model.predict """
    with ExeTimer() as self._internal_timer:
      self.offset = self.acc_energy
      if enable_earlyexit:
        self._enable_earlyexit = True
        self._toggle_internal_early_exit()
      if eager_predict:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()
      outputs = super().predict(x, batch_size, verbose, steps,
                                callbacks, max_queue_size, workers,
                                use_multiprocessing) if use_wrapper else self(x)
    if enable_earlyexit:
      self._toggle_internal_early_exit()
    if eager_predict:
      tf.config.run_functions_eagerly(False)
    return outputs
  def predict_partial(
    self,
    x,
    batch_size=None,
    verbose="auto",
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    run_until=-1,
  ):
    """ Function override for keras.layers.Model.predict with partial execution """
    def step_function(model, iterator):
      def run_step(data):
        outputs = model.predict_step(data)
        model.predict_counter.assign_add(1)
        return outputs
      data = next(iterator)
      outputs = model.distribute_strategy.run(run_step, args=(data,))
      outputs = reduce_per_replica(outputs, self.distribute_strategy, reduction="concat")
      return outputs
    def predict_function(iterator):
      return step_function(self, iterator)
    self.predict_function = tf.function(predict_function, reduce_tracing=True)
    self.exec_mode = ExecutionMode.PARTIAL_RUN
    self.run_until = run_until
    outputs = super().predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)
    self.exec_mode = ExecutionMode.NORMAL_EXEC
    return outputs
  def profile(self, x, batch_size: int = 32, use_wrapper: bool = True,
                    base_path: str = "", eager_predict: bool = True,
                    profile_mode: ProfileMode = ProfileMode.FULL):
    """ Defines the logic to profile the model """
    self.exec_mode = ExecutionMode.PROFILE
    self.profile_mode = profile_mode
    while True:
      try:
        self.predict_function = None
        self.predict(x, batch_size, verbose= 0, eager_predict=eager_predict, use_wrapper=use_wrapper)
        if self.jetson_stats:
          if not self.jetson_stats._started.is_set():
            self.jetson_stats.start()
          else:
            self.jetson_stats.close()
            self.jetson_stats =  jtop(interval=0.1)
            self.jetson_stats.start()
          self.jetson_stats.attach(self.accumulate_power)
          self.last_time = time.time()
          self.acc_energy = 0
        self.predict(x, batch_size, verbose= 0, eager_predict=eager_predict, use_wrapper=use_wrapper)
        if self.jetson_stats:
          self.jetson_stats.detach(self.accumulate_power)
          self.jetson_stats.close()
          self.acc_energy = 0
        self.predict_function = None
      except tf.errors.ResourceExhaustedError:
        if not use_wrapper:
          print("Defaulting to TF to manage memory...")
          use_wrapper = True
        else:
          new_batch =  batch_size // 2
          batch_size = new_batch if new_batch > 0 else 1
          print(f"Reducing batch size to {batch_size} due to OOM...")
        continue
      break
    os.makedirs(base_path, exist_ok=True)
    profile_data_path = os.path.join(base_path, f"profile_data_{profile_mode.name.lower()}.pickle")
    with open(profile_data_path, "wb") as file_handle:
      pickle.dump(self.profile_data, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    del self.jetson_stats
    self.jetson_stats = None
  def _run_internal_graph(self, inputs, training=None, mask=None):
    inputs = self._flatten_to_reference_inputs(inputs)
    if mask is None:
      mask = [None] * len(inputs)
    else:
      mask = self._flatten_to_reference_inputs(mask)
    for input_tensor, input_mask in zip(inputs, mask):
      input_tensor._keras_mask = input_mask
    tensor_dict = dict()
    tensor_usage_count = self._tensor_usage_count
    for input_node, input_t in zip(self.inputs, inputs):
      input_t = self._conform_to_reference_input(input_t, ref_input=input_node)
      in_id = str(id(input_node))
      tensor_dict[in_id] = [input_t] * tensor_usage_count[in_id]
    nodes_by_depth = self._nodes_by_depth
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    curr_layer = ""
    try:
      for depth in depth_keys:
        nodes = nodes_by_depth[depth]
        for node in nodes:
          if node.is_input:
            continue  # Input tensors already exist.

          if any(t_id not in tensor_dict for t_id in node.flat_input_ids):
            continue  # Node is not computable, try skipping.

          args, kwargs = node.map_arguments(tensor_dict)

          outputs = node.layer(*args, **kwargs)
          
          curr_layer = node.layer.name

          if isinstance(node.layer, EarlyExit) and self.run_eagerly:
            if self._enable_earlyexit:
              if node.layer.predict and self.evaluator(outputs[-1], self.thr):
                for x, y in zip(self.outputs, cycle(tf.nest.flatten(outputs[-1]))):
                  x_id = str(id(x))
                  tensor_dict[x_id] = [y] * \
                      tensor_usage_count[x_id]
                raise EarlyExitException(outputs[-1])
          # Update tensor_dict.
          for x_id, y in zip(
            node.flat_output_ids, tf.nest.flatten(outputs)
          ):
            tensor_dict[x_id] = [y] * tensor_usage_count[x_id]
    except EarlyExitException as e:
      # This is handle cleanly since by using eager mode and 
      # direct __call__, we avoid the predict() function.
      self.exit_activated = True
      self.layer_exit = curr_layer
      return e.value.numpy()

    if self._internal_timer and curr_layer != "":
      self._internal_timer.visited_layers.append(curr_layer)
    output_tensors = []
    for x in self.outputs:
      x_id = str(id(x))
      assert x_id in tensor_dict, "Could not compute output " + str(x)
      output_tensors.append(tensor_dict[x_id].pop())
    return tf.nest.pack_sequence_as(self._nested_outputs, output_tensors)
  def _profile_internal_graph(self, inputs, training=None, mask=None):
    """ Defines the logic to profile the model graph """
    inputs = self._flatten_to_reference_inputs(inputs)
    if mask is None:
      mask = [None] * len(inputs)
    else:
      mask = self._flatten_to_reference_inputs(mask)
    for input_tensor, input_mask in zip(inputs, mask):
      input_tensor._keras_mask = input_mask
    tensor_dict = dict()
    tensor_usage_count = self._tensor_usage_count
    for input_node, input_t in zip(self.inputs, inputs):
      input_t = self._conform_to_reference_input(input_t, ref_input=input_node)
      in_id = str(id(input_node))
      tensor_dict[in_id] = [input_t] * tensor_usage_count[in_id]
    nodes_by_depth = self._nodes_by_depth
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    tensor_dict = self._traverse_internal_graph(tensor_dict, depth_keys,
                                                nodes_by_depth, tensor_usage_count)
    output_tensors = []
    for out_layer in self.outputs:
      out_id = str(id(out_layer))
      assert out_id in tensor_dict, f"Could not compute output {out_layer}"
      output_tensors.append(tensor_dict[out_id].pop())
    return tf.nest.pack_sequence_as(self.outputs, output_tensors)
  def _run_partial_graph(self, inputs, training=None, mask=None):
    """ Defines the logic to run the model graph partially """
    inputs = self._flatten_to_reference_inputs(inputs)
    if mask is None:
      mask = [None] * len(inputs)
    else:
      mask = self._flatten_to_reference_inputs(mask)
    for input_tensor, input_mask in zip(inputs, mask):
      input_tensor._keras_mask = input_mask
    tensor_dict = dict()
    tensor_usage_count = self._tensor_usage_count
    for input_node, input_t in zip(self.inputs, inputs):
      input_t = self._conform_to_reference_input(input_t, ref_input=input_node)
      in_id = str(id(input_node))
      tensor_dict[in_id] = [input_t] * tensor_usage_count[in_id]
    nodes_by_depth = self._nodes_by_depth
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    idx = self.run_until if self.run_until > 0 and \
                            self.run_until < len(depth_keys) \
                          else len(depth_keys)-1
    depth_keys = depth_keys[:idx]
    tensor_dict = self._traverse_internal_graph(tensor_dict, depth_keys,
                                                nodes_by_depth, tensor_usage_count)
    nodes = nodes_by_depth[depth_keys[-1]]
    output_tensors = []
    nested_outputs = []
    for node in nodes:
      nested_outputs.append(node.layer.output)
      for x_id in node.flat_output_ids:
        assert x_id in tensor_dict, "Could not compute output " + x_id
        output_tensors.append(tensor_dict[x_id].pop())
    if len(nested_outputs) == 1:
      return tf.nest.pack_sequence_as(nested_outputs[0], output_tensors)
    return tf.nest.pack_sequence_as(nested_outputs, output_tensors)
  def _traverse_internal_graph(self, tensor_dict, depth_keys, nodes_by_depth, tensor_usage_count):
    """ Global logic to traverse model graph """
    total_exec_time_depth = total_exec_time_layer = acc_mem_size = 0
    for depth_idx, depth in enumerate(depth_keys, start=0):
      nodes =  nodes_by_depth[depth]
      with ExeTimer() as depth_timer:
        for node in nodes:
          if node.is_input:
            continue # We ignore an already existing tensor
          if any(t_id not in tensor_dict for t_id in node.flat_input_ids):
            continue # Node cannot be computed... We try skipping it
          args, kwargs = node.map_arguments(tensor_dict)
          with ExeTimer() as node_timer:
            outputs = node.layer(*args, **kwargs)
          if self.profile_mode == ProfileMode.LAYER_ONLY:
            mem_req = layer_profiler(node.layer)
            acc_mem_size += mem_req
            total_exec_time_layer += node_timer.execution
            self.profile_data["layer_data"].update({
              node.layer.name: collect_metrics( mem_size=mem_req,
                                                acc_time=total_exec_time_layer,
                                                **{'energy': self.acc_energy} if self.jetson_stats else {})})
            self.profile_data["depth_data"].update({
              depth_idx: collect_metrics(acc_mem_size=acc_mem_size)
            })
          if isinstance(node.layer, EarlyExit) and self.run_eagerly:
            if node.layer.predict and is_high_watermark(outputs[-1], self.thr):
              for out_layer, output in zip(self.outputs, cycle(tf.nest.flatten(outputs[-1]))):
                out_id = str(id(out_layer))
                tensor_dict[out_id] = [output] * tensor_usage_count[out_id]
          # Update the tensor dict in any other case
          for x_id, y in zip(
              node.flat_output_ids,
              tf.nest.flatten(outputs)
          ):
            tensor_dict[x_id] = [y] * tensor_usage_count[x_id]
      if self.profile_mode.FULL or self.profile_mode.DEPTH_ONLY:
        total_exec_time_depth += depth_timer.execution
        if depth_idx in self.profile_data["depth_data"]:
          self.profile_data["depth_data"][depth_idx].update(
            collect_metrics(time=depth_timer.execution, acc_time=self._internal_timer.current(),
                            acc_depth_time=total_exec_time_depth,
                            **{"energy": (self.acc_energy-self.offset)/3600, "power": self.power} if self.jetson_stats else {})
          )
    return tensor_dict
def is_high_watermark(outputs, thr: float = 0.5):
  """ Returns true if the outputs are above the threshold """
  max_by_row = tf.math.reduce_max(outputs, axis=1)
  if max_by_row.shape[0] == 1:
    return max_by_row[0] >= thr
  return tf.math.reduce_mean(max_by_row) >= thr
def reduce_per_replica(values, strategy, reduction):
  if reduction == "auto":
      reduction = "first" if backend.is_tpu_strategy(strategy) else "sum"
  def _reduce(v):
    """Reduce a single `PerReplica` object."""
    if _collective_all_reduce_multi_worker(strategy):
      if reduction == "concat":
        return _multi_worker_concat(v, strategy)
      elif reduction == "sum":
        return strategy.reduce("SUM", v, axis=None)

    if not _is_per_replica_instance(v):
      return v
    elif reduction == "first":
      return strategy.experimental_local_results(v)[0]
    elif reduction == "concat":
      if _is_tpu_multi_host(strategy):
        return _tpu_multi_host_concat(v, strategy)
      else:
        return concat(strategy.experimental_local_results(v))
    elif reduction == "sum":
      return tf.reduce_sum(strategy.experimental_local_results(v))
    else:
      raise ValueError(
        '`reduction` must be "first", "concat", "sum", or "auto". '
        f"Received: reduction={reduction}."
      )

  return tf.nest.map_structure(_reduce, values)
def _collective_all_reduce_multi_worker(strategy):
  return (
    isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy)
  ) and strategy.extended._in_multi_worker_mode()
def _multi_worker_concat(v, strategy):
  """Order PerReplica objects for CollectiveAllReduceStrategy and concat."""
  replicas = strategy.gather(v, axis=0)
  # v might not have the same shape on different replicas
  if _is_per_replica_instance(v):
    shapes = tf.concat(
      [
          tf.expand_dims(tf.shape(single_value)[0], axis=0)
          for single_value in v.values
      ],
      axis=0,
    )
    all_shapes = strategy.gather(shapes, axis=0)
  else:
    # v is a tensor. This may happen when, say, we have 2x1 multi-worker.
    all_shapes = strategy.gather(
      tf.expand_dims(tf.shape(v)[0], axis=0), axis=0
    )

  replicas = tf.split(
    replicas,
    num_or_size_splits=all_shapes,
    num=strategy.num_replicas_in_sync,
  )
  ordered_replicas = []
  num_replicas_per_worker = len(strategy.extended.worker_devices)
  for replica_id in range(num_replicas_per_worker):
    ordered_replicas += replicas[replica_id::num_replicas_per_worker]
  return concat(ordered_replicas)
def _is_per_replica_instance(obj):
  return isinstance(obj, tf.distribute.DistributedValues) and isinstance(
    obj, tf.__internal__.CompositeTensor
  )
def concat(tensors, axis=0):
  """Concats `tensor`s along `axis`."""
  if isinstance(tensors[0], tf.SparseTensor):
    return tf.sparse.concat(axis=axis, sp_inputs=tensors)
  elif _is_scalar(tensors[0]):
    return tf.stack(tensors, axis=axis)
  else:
    return tf.concat(tensors, axis=axis)
def _is_scalar(x):
  return isinstance(x, (tf.Tensor, tf.Variable)) and x.shape.rank == 0
def _is_tpu_multi_host(strategy):
  return backend.is_tpu_strategy(strategy) and strategy.extended.num_hosts > 1
def _tpu_multi_host_concat(v, strategy):
  """Correctly order TPU PerReplica objects."""
  replicas = strategy.experimental_local_results(v)
  # When distributed datasets are created from Tensors / NumPy,
  # TPUStrategy.experimental_distribute_dataset shards data in
  # (Replica, Host) order, and TPUStrategy.experimental_local_results returns
  # it in (Host, Replica) order.
  # TODO(b/150317897): Figure out long-term plan here.
  num_replicas_per_host = strategy.extended.num_replicas_per_host
  ordered_replicas = []
  for replica_id in range(num_replicas_per_host):
    ordered_replicas += replicas[replica_id::num_replicas_per_host]
  return concat(ordered_replicas)
