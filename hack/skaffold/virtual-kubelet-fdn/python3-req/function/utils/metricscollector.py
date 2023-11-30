""" Generates a dictionary with the required metrics """
import copy
from typing import Union, Dict
import numpy as np
from pkg_resources import working_set
METRICS = ['flops', 'time', 'data_output', 'mem_size', 'acc_time', 'energy', 'acc_mem_size', 'power', "acc_depth_time"]

installed_packages = [pkg.key for pkg in working_set]

def collect_metrics(**kwargs):
  """ Collects the metrics from the model """
  metrics = copy.deepcopy(METRICS)
  if 'jetson-stats' not in installed_packages:
    metrics.pop(metrics.index('power'))
    metrics.pop(metrics.index('energy'))
  metrics_dict = {}
  for metric in metrics:
    if metric in kwargs:
      metrics_dict[metric] = kwargs.get(metric, None)
  del metrics
  return metrics_dict
def compute_accuracies(model, model_pred, data_true, thr = 0.5):
  """ Computes the accuracies for the model """
  def compute_acc_tr(outputs, labels):
    thr_comply = correct = 0
    for y_pred, y_true in zip(outputs, labels):
      if np.max(y_pred) >= thr:
        thr_comply += 1
      if np.argmax(y_pred) == np.argmax(y_true):
        correct += 1
    return correct/len(outputs), thr_comply/len(outputs)
  results: Dict[str, Union[str, float]] = {"thr": thr}
  for output_node, output in zip(model.outputs, model_pred):
    layer_name = output_node.name.layer.name
    tr, acc = compute_acc_tr(output, data_true)
    results.update({
      f"{layer_name}_tr": tr,
      f"{layer_name}_acc": acc
    })
  if model._internal_timer:
    results.update({"exec_time": model._internal_timer.execution})
    if len(model._internal_timer.visited_layers) > 0:
      concat_layers: str = ",".join(model._internal_timer.visited_layers)
      results.update({"output_layers": concat_layers})
