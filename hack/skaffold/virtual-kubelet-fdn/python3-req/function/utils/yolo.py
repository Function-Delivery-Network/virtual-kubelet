""" 
Module describing YOLOv3 model 
Source code from: https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/models.py
Adapted for use in this project
"""
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from keras import layers
from keras import regularizers
from keras.models import Model

from official.modeling.optimization.lr_schedule import LinearWarmup
from keras.optimizer_v2 import adam, learning_rate_schedule
from keras import optimizers
from keras import backend as K
from keras.losses import (
  binary_crossentropy,
  sparse_categorical_crossentropy
)

from backend.model import DistributedModel
from backend.earlyexit import EarlyExit
from utils.coco_preprocessor import data_generator
from utils import coco_preprocessor
yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32)

yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32)
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

def darknet_conv_block(x, filters, size, strides = 1, batch_norm = True):
  """ Defines a Darknet Conv Block"""
  if strides == 1:
    padding = 'same'
  else:
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    padding = 'valid'
  x = layers.Conv2D(
    filters = filters,
    kernel_size=size,
    strides=strides,
    padding=padding,
    use_bias=not batch_norm,
    kernel_regularizer=regularizers.l2(0.0005),
  )(x)
  if batch_norm:
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
  return x

def darknet_residual_block(x, filters):
  """ Defines a Darknet Residual Block"""
  prev = x
  x = darknet_conv_block(x, filters // 2, 1)
  x = darknet_conv_block(x, filters, 3)
  x = layers.Add()([prev, x])
  return x

def darknet_block(x, filters, blocks):
  """ Defines a Darknet Block"""
  x = darknet_conv_block(x, filters, 3, strides=2)
  for _ in range(blocks):
    x = darknet_residual_block(x, filters)
  return x

def darknet(x_in):
  """ Defines the Darknet-61 architecture"""
  x = darknet_conv_block(x_in, 32, 3)
  x = darknet_block(x, 64, 1)
  x = darknet_block(x, 128, 2)
  x = x_36 = darknet_block(x, 256, 8)
  x = x_61 = darknet_block(x, 512, 8)
  x = darknet_block(x, 1024, 4)
  return x_36, x_61, x

def darknet_tiny(x_in):
  """ Defines a Darknet-tiny architecture"""
  x = darknet_conv_block(x_in, 16, 3)
  x = layers.MaxPool2D(2, 2, 'same')(x)
  x = darknet_conv_block(x, 32, 3)
  x = layers.MaxPool2D(2, 2, 'same')(x)
  x = darknet_conv_block(x, 64, 3)
  x = layers.MaxPool2D(2, 2, 'same')(x)
  x = darknet_conv_block(x, 128, 3)
  x = layers.MaxPool2D(2, 2, 'same')(x)
  x = x_8 = darknet_conv_block(x, 256, 3)
  x = layers.MaxPool2D(2, 2, 'same')(x)
  x = darknet_conv_block(x, 512, 3)
  x = layers.MaxPool2D(2, 1, 'same')(x)
  x = darknet_conv_block(x, 1024, 3)
  return x_8, x

def yolo_conv_block(x_in, filters, name=None):
  """ Defines a YOLO Conv Block"""
  if isinstance(x_in, tuple):
    inputs = x_in[0], x_in[1]
    x, x_skip = inputs
    x = darknet_conv_block(x, filters, 1)
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, x_skip])
  else:
    x = x_in
  x = darknet_conv_block(x, filters, 1)
  x = darknet_conv_block(x, filters * 2, 3)
  x = darknet_conv_block(x, filters, 1)
  x = darknet_conv_block(x, filters * 2, 3)
  x = darknet_conv_block(x, filters, 1)
  return x
  

def yolo_conv_tiny(x_in, filters, name = None):
  """ Defines a YOLO Conv Block for the Tiny model"""
  if isinstance(x_in, tuple):
    x, x_skip = x_in[0], x_in[1]
    x = darknet_conv_block(x, filters, 1)
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, x_skip])
  else:
    x = darknet_conv_block(x_in, filters, 1)
  return x

def yolo_ouput(x_in, filters, anchors, classes, name = None):
  x = darknet_conv_block(x_in, filters * 2, 3)
  x = darknet_conv_block(x, anchors * (classes + 5), 1, batch_norm=False)
  x = layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)), name=name)(x)
  return x

def yolo_v3(size_x,size_y, channels=3,
                anchors = yolo_anchors,
                masks = yolo_anchor_masks,
                classes = 80,
                training=False):
  """ Defines the YOLOv3 Tiny architecture"""
  x = inputs = layers.Input([None, None, channels])
  x_36, x_61, x = darknet(x)
  x = yolo_conv_block(x, 512, name='yolo_conv_0')
  output_0 = yolo_ouput(x, 512, len(masks[0]), classes, name='yolo_output_0')
  x = yolo_conv_block((x, x_61), 256, name='yolo_conv_1')
  output_1 = yolo_ouput(x, 256, len(masks[1]), classes, name='yolo_output_1')
  x = yolo_conv_block((x, x_36), 128, name='yolo_conv_1')
  output_2 = yolo_ouput(x, 128, len(masks[2]), classes, name='yolo_output_1')
  return DistributedModel(inputs=inputs, outputs=(output_0, output_1, output_2), name='yolov3_tiny')

def _meshgrid(n_a, n_b):

  return [
      tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
      tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
  ]

def yolo_boxes(pred, anchors, classes):
  # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
  grid_size = tf.shape(pred)[1:3]
  box_xy, box_wh, objectness, class_probs = tf.split(
      pred, (2, 2, 1, classes), axis=-1)

  box_xy = tf.sigmoid(box_xy)
  objectness = tf.sigmoid(objectness)
  class_probs = tf.sigmoid(class_probs)
  pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

  # !!! grid[x][y] == (y, x)
  grid = _meshgrid(grid_size[1],grid_size[0])
  grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

  box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
      tf.cast(grid_size, tf.float32)
  box_wh = tf.exp(box_wh) * anchors

  box_x1y1 = box_xy - box_wh / 2
  box_x2y2 = box_xy + box_wh / 2
  bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

  return bbox, objectness, class_probs, pred_box

def broadcast_iou(box_1, box_2):
  # box_1: (..., (x1, y1, x2, y2))
  # box_2: (N, (x1, y1, x2, y2))

  # broadcast boxes
  box_1 = tf.expand_dims(box_1, -2)
  box_2 = tf.expand_dims(box_2, 0)
  # new_shape: (..., N, (x1, y1, x2, y2))
  new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
  box_1 = tf.broadcast_to(box_1, new_shape)
  box_2 = tf.broadcast_to(box_2, new_shape)

  int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                      tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
  int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                      tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
  int_area = int_w * int_h
  box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
      (box_1[..., 3] - box_1[..., 1])
  box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
      (box_2[..., 3] - box_2[..., 1])
  return int_area / (box_1_area + box_2_area - int_area)

# def yolo_v3_loss(anchors, classes=80, ignore_thresh=0.5):
#   def yolo_loss(y_true, y_pred):
#     pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
#         y_pred, anchors, classes)
#     pred_xy = pred_xywh[..., 0:2]
#     pred_wh = pred_xywh[..., 2:4]
#     print(f"VALUE TO SPLIT: {y_true} vs {y_pred}")
#     true_box, true_obj, true_class_idx = tf.split(
#         y_true, (4, 1, 1), axis=-1)
#     true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
#     true_wh = true_box[..., 2:4] - true_box[..., 0:2]

#     # give higher weights to small boxes
#     box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]
    
#     grid_size = tf.shape(y_true)[1]
#     grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
#     grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
#     true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
#         tf.cast(grid, tf.float32)
#     true_wh = tf.math.log(true_wh / anchors)
#     true_wh = tf.where(tf.math.is_inf(true_wh),
#                         tf.zeros_like(true_wh), true_wh)

#     # 4. calculate all masks
#     obj_mask = tf.squeeze(true_obj, -1)
#     # ignore false positive when iou is over threshold
#     best_iou = tf.map_fn(
#         lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
#             x[1], tf.cast(x[2], tf.bool))), axis=-1),
#         (pred_box, true_box, obj_mask),
#         tf.float32)
#     ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

#     # 5. calculate all losses
#     xy_loss = obj_mask * box_loss_scale * \
#         tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
#     wh_loss = obj_mask * box_loss_scale * \
#         tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
#     obj_loss = binary_crossentropy(true_obj, pred_obj)
#     obj_loss = obj_mask * obj_loss + \
#         (1 - obj_mask) * ignore_mask * obj_loss
#     # TODO: use binary_crossentropy instead
#     class_loss = obj_mask * sparse_categorical_crossentropy(
#         true_class_idx, pred_class)

#     # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
#     xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
#     wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
#     obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
#     class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

#     return xy_loss + wh_loss + obj_loss + class_loss
#   return yolo_loss


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def _process_feats(out, anchors, mask):
  """process output features.

  # Arguments
      out: Tensor (N, N, 3, 4 + 1 +80), output feature map of yolo.
      anchors: List, anchors for box.
      mask: List, mask for anchors.

  # Returns
      boxes: ndarray (N, N, 3, 4), x,y,w,h for per box.
      box_confidence: ndarray (N, N, 3, 1), confidence for per box.
      box_class_probs: ndarray (N, N, 3, 80), class probs for per box.
  """
  grid_h, grid_w, num_boxes = map(int, out.shape[0:3])

  # anchors = [anchors[i] for i in mask]
  # Reshape to batch, height, width, num_anchors, box_params.
  anchors_tensor = K.reshape(K.variable(anchors),
                              [1, 1, len(anchors), 2])
  
  box_xy = K.get_value(K.sigmoid(out[..., :2]))
  box_wh = K.get_value(K.exp(out[..., 2:4]) * anchors_tensor)
  box_confidence = K.get_value(K.sigmoid(out[..., 4]))
  box_confidence = np.expand_dims(box_confidence, axis=-1)
  box_class_probs = K.get_value(K.sigmoid(out[..., 5:]))

  col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
  row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

  col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
  row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
  grid = np.concatenate((col, row), axis=-1)

  box_xy += grid
  box_xy /= (grid_w, grid_h)
  box_wh /= (416, 416)
  box_xy -= (box_wh / 2.)
  boxes = np.concatenate((box_xy, box_wh), axis=-1)

  return boxes, box_confidence, box_class_probs

def _filter_boxes(boxes, box_confidences, box_class_probs):
  """Filter boxes with object threshold.

  # Arguments
      boxes: ndarray, boxes of objects.
      box_confidences: ndarray, confidences of objects.
      box_class_probs: ndarray, class_probs of objects.

  # Returns
      boxes: ndarray, filtered boxes.
      classes: ndarray, classes for boxes.
      scores: ndarray, scores for boxes.
  """
  box_scores = box_confidences * box_class_probs
  box_classes = np.argmax(box_scores, axis=-1)
  box_class_scores = np.max(box_scores, axis=-1)
  pos = np.where(box_class_scores >= 0.7)

  boxes = boxes[pos]
  classes = box_classes[pos]
  scores = box_class_scores[pos]

  return boxes, classes, scores


def _nms_boxes(boxes, scores):
  """Suppress non-maximal boxes.

  # Arguments
      boxes: ndarray, boxes of objects.
      scores: ndarray, scores of objects.

  # Returns
      keep: ndarray, index of effective boxes.
  """
  x = boxes[:, 0]
  y = boxes[:, 1]
  w = boxes[:, 2]
  h = boxes[:, 3]

  areas = w * h
  order = scores.argsort()[::-1]

  keep = []
  while order.size > 0:
      i = order[0]
      keep.append(i)

      xx1 = np.maximum(x[i], x[order[1:]])
      yy1 = np.maximum(y[i], y[order[1:]])
      xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
      yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

      w1 = np.maximum(0.0, xx2 - xx1 + 1)
      h1 = np.maximum(0.0, yy2 - yy1 + 1)
      inter = w1 * h1

      ovr = inter / (areas[i] + areas[order[1:]] - inter)
      inds = np.where(ovr <= 0.7)[0]
      order = order[inds + 1]

  keep = np.array(keep)

  return keep

def yolo_loss(anchors, num_classes, ignore_thresh=.5, print_loss=False, index=0):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    def yolo_fn(y_true, y_out):
      num_layers = 1 # default setting
      yolo_outputs = y_out
      anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
      input_shape = K.cast(K.shape(yolo_outputs)[1:3] * 8, K.dtype(y_true[0]))
      grid_shapes = K.cast(K.shape(yolo_outputs)[1:3], K.dtype(y_true[0]))
      loss = 0
      m = K.shape(yolo_outputs)[0] # batch size, tensor
      mf = K.cast(m, K.dtype(yolo_outputs))

      object_mask = y_true[..., 4:5]
      true_class_probs = y_true[..., 5:]

      grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs,
        anchors, num_classes, input_shape, calc_loss=True)
      pred_box = K.concatenate([pred_xy, pred_wh])
      
      
      # Darknet raw box to calculate loss.
      raw_true_xy = y_true[..., :2]*grid_shapes[::-1] - grid
      raw_true_wh = K.log(y_true[..., 2:4] / anchors * input_shape[::-1])
      # print("raw_true_wh", raw_true_wh)
      raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
      box_loss_scale = 2 - y_true[...,2:3]*y_true[...,3:4]
      
      # Find ignore mask, iterate over each of batch.
      ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
      object_mask_bool = K.cast(object_mask, 'bool')
      def loop_body(b, ignore_mask):
        true_box = tf.boolean_mask(y_true[b,...,0:4], object_mask_bool[b,...,0])
        iou = box_iou(pred_box[b], true_box)
        best_iou = K.max(iou, axis=-1)
        ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
        return b+1, ignore_mask
      _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
      ignore_mask = ignore_mask.stack()
      ignore_mask = K.expand_dims(ignore_mask, -1)
      
      # K.binary_crossentropy is helpful to avoid exp overflow.
      xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
      wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
      confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
        (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
      class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)
      
      xy_loss = K.sum(xy_loss) / mf
      wh_loss = K.sum(wh_loss) / mf
      confidence_loss = K.sum(confidence_loss) / mf
      class_loss = K.sum(class_loss) / mf
      loss += xy_loss + wh_loss + confidence_loss + class_loss
      # if print_loss:
      #     batch_score = []
      #     for idx in range(m):
      #       scores = []
      #       classes = []
      #       boxes = []
      #       a,b,c = _process_feats(yolo_outputs[idx], anchors, anchors)
      #       a,b,c = _filter_boxes(a,b,c)
      #       scores.append(c)
      #       classes.append(b)
      #       boxes.append(a)

      #       boxes = np.concatenate(boxes)
      #       classes = np.concatenate(classes)
      #       scores = np.concatenate(scores)
            
      #       nboxes, nclasses, nscores = [], [], []
      #       for c in set(classes):
      #         inds = np.where(classes ==c)
      #         b = boxes[inds]
      #         s = scores[inds]
      #         c = classes[inds]
      #         keep = _nms_boxes(b, s)
      #         nboxes.append(b[keep])
      #         nclasses.append(c[keep])
      #         nscores.append(s[keep])
      #       if len(nscores)>0:
      #         nscores = np.concatenate(nscores)
      #         batch_score.append(tf.reduce_max(nscores))
      #     tf.print("- ", " prob:", tf.reduce_max(batch_score))
      return loss
    return yolo_fn

def yolo_v3_input_processor(inputs):
  return tf.expand_dims(inputs['image'], axis=0), tf.stack([inputs['objects']['bbox'],tf.cast(inputs['objects']['label'],tf.float32)], axis=1)

def yolo_v3_tiny_ee(x_in, filters, anchors, classes, name=None):
  x = darknet_conv_block(x_in, filters*2, 3)
  x = darknet_conv_block(x, anchors * (classes + 5), 1, batch_norm=False)
  x = layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
  x, out = EarlyExit(inputs=[x_in, x], name=name)([x_in, x])
  return x, out

def yolo_v3_tiny(size_x,size_y, channels=3,
                anchors = yolo_tiny_anchors,
                masks = yolo_tiny_anchor_masks,
                classes = 80,
                training=False):
  """ Defines the YOLOv3 Tiny architecture"""
  x = inputs = layers.Input([None, None, channels])
  x_8, x = darknet_tiny(x)
  x, output_2 = yolo_v3_tiny_ee(x, 256, len(masks[0]), classes, name='yolo_output_2')
  x = yolo_conv_tiny(x, 256, name='yolo_conv_0')
  output_0 = yolo_ouput(x, 256, len(masks[0]), classes, name='yolo_output_0')
  x = yolo_conv_tiny((x, x_8), 128, name='yolo_conv_1')
  x, output_3 = yolo_v3_tiny_ee(x, 128, len(masks[1]), classes, name='yolo_output_3')
  output_1 = yolo_ouput(x, 128, len(masks[1]), classes, name='yolo_output_1')
  model = DistributedModel(
    inputs=inputs, 
    outputs=[output_0, output_1, output_2, output_3],
    name='yolov3_tiny')
  return model

  h, w = 416, 416
  num_anchors = len(anchors)
  y_true = [layers.Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, classes+5)) for l in range(2)]
  
  model_loss = layers.Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': classes, 'ignore_thresh': 0.7})(
        [*model.output, *y_true])
  model = Model([model.input, *y_true], model_loss)
  return model

def train_yolo_v3_tiny(model, epochs=10, anchors=yolo_tiny_anchors, classes=80):


  with open("2012_train.txt") as file:
    train_files = file.readlines()

  with open("2012_val.txt") as file:
    val_files = file.readlines()

  h, w = 416, 416
  num_anchors = len(anchors)
  # y_true = [layers.Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
  #       num_anchors//2, classes+5)) for l in range(2)]
  
  # model_loss = layers.Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
  #       arguments={'anchors': anchors, 'num_classes': classes, 'ignore_thresh': 0.7})(
  #       [*model.output, *y_true])
  # model = Model([model.input, *y_true], model_loss)

  scheduler = learning_rate_schedule.ExponentialDecay(
    initial_learning_rate=1e-3, 
    decay_steps=500,
    decay_rate=0.96,
  )

  loss = [yolo_loss(anchors[mask], num_classes=classes, print_loss=True, index=idx, ignore_thresh=0.7) for idx, mask in enumerate(yolo_tiny_anchor_masks)]
  loss = [*loss, *loss]
  optimizer = optimizers.nadam_v2.Nadam(
    learning_rate=0.001,
    decay=0.0005,
  )
  model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)

  batch_size = 16
  len_files = len(train_files)
  len_val = len(val_files)
  model.fit(data_generator(train_files, batch_size, (416, 416), yolo_tiny_anchors, classes)(),
    epochs=epochs,
    steps_per_epoch=max(1, len_files//batch_size),
    validation_steps=max(1, len_val//batch_size),
    validation_data=data_generator(val_files, batch_size, (416, 416), yolo_tiny_anchors, classes)(),
  )

