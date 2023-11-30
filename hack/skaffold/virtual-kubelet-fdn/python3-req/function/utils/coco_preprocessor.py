""" COCO Dataset Preprocessor

    It formats the COCO dataset as the YOLOv3
    model format for training and evaluation.

    Original implementation: https://github.com/qqwweee/keras-yolo3
"""
import tensorflow as tf
import numpy as np
import json
from collections import defaultdict
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import xml.etree.ElementTree as ET

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32)

def generate_train_annotations():
  """ Generates the annotations file for the COCO dataset """
  name_box_id = defaultdict(list)
  id_name = dict()
  with open('/u/home/nui/annotations/instances_train2017.json', encoding='utf-8') as file:
    data = json.load(file)
  
  annotations = data['annotations']

  for annon in annotations:
    id = annon['image_id']
    name = '/u/home/nui/train2017/%012d.jpg' % id
    cat = annon['category_id']

    if cat >= 1 and cat <= 11:
      cat = cat - 1
    elif cat >= 13 and cat <= 25:
      cat = cat - 2
    elif cat >= 27 and cat <= 28:
      cat = cat - 3
    elif cat >= 31 and cat <= 44:
      cat = cat - 5
    elif cat >= 46 and cat <= 65:
      cat = cat - 6
    elif cat == 67:
      cat = cat - 7
    elif cat == 70:
      cat = cat - 9
    elif cat >= 72 and cat <= 82:
      cat = cat - 10
    elif cat >= 84 and cat <= 90:
      cat = cat - 11
    
    name_box_id[name].append([annon['bbox'], cat])

  with open('train.txt', 'w') as file:
    for key in name_box_id.keys():
      file.write(key)
      box_infos = name_box_id[key]
      for info in box_infos:
        x_min = int(info[0][0])
        y_min = int(info[0][1])
        x_max = x_min + int(info[0][2])
        y_max = y_min + int(info[0][3])

        box_info = " %d,%d,%d,%d,%d" % (
            x_min, y_min, x_max, y_max, int(info[1]))
        file.write(box_info)
      file.write('\n')

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annon_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
  line = annon_line.split()
  image = Image.open(line[0])
  iw, ih = image.size
  h, w = input_shape
  box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

  if not random:
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2
    image_data = 0
    if proc_img:
      image = image.resize((nw, nh), Image.BICUBIC)
      new_image = Image.new('RGB', (w, h), (128, 128, 128))
      new_image.paste(image, (dx, dy))
      image_data = np.array(new_image)/255

    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
      np.random.shuffle(box)
      if len(box) > max_boxes:
        box = box[:max_boxes]
      box[:, [0, 2]] = box[:, [0, 2]]*scale + dx
      box[:, [1, 3]] = box[:, [1, 3]]*scale + dy
      box_data[:len(box)] = box

    return image_data, box_data
  new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
  scale = rand(.25, 2)
  if new_ar < 1:
      nh = int(scale*h)
      nw = int(nh*new_ar)
  else:
      nw = int(scale*w)
      nh = int(nw/new_ar)
  image = image.resize((nw,nh), Image.BICUBIC)

  # place image
  dx = int(rand(0, w-nw))
  dy = int(rand(0, h-nh))
  new_image = Image.new('RGB', (w,h), (128,128,128))
  new_image.paste(image, (dx, dy))
  image = new_image

  # flip image or not
  flip = rand()<.5
  if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

  # distort image
  hue = rand(-hue, hue)
  sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
  val = rand(1, val) if rand()<.5 else 1/rand(1, val)
  x = rgb_to_hsv(np.array(image)/255.)
  x[..., 0] += hue
  x[..., 0][x[..., 0]>1] -= 1
  x[..., 0][x[..., 0]<0] += 1
  x[..., 1] *= sat
  x[..., 2] *= val
  x[x>1] = 1
  x[x<0] = 0
  image_data = hsv_to_rgb(x) # numpy array, 0 to 1

  # correct boxes
  box_data = np.zeros((max_boxes,5))
  if len(box)>0:
    np.random.shuffle(box)
    box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
    box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
    if flip: box[:, [0,2]] = w - box[:, [2,0]]
    box[:, 0:2][box[:, 0:2]<0] = 0
    box[:, 2][box[:, 2]>w] = w
    box[:, 3][box[:, 3]>h] = h
    box_w = box[:, 2] - box[:, 0]
    box_h = box[:, 3] - box[:, 1]
    box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
    if len(box)>max_boxes: box = box[:max_boxes]
    box_data[:len(box)] = box

  return image_data, box_data

def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
  '''Preprocess true boxes to training input format

  Parameters
  ----------
  true_boxes: array, shape=(m, T, 5)
      Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
  input_shape: array-like, hw, multiples of 32
  anchors: array, shape=(N, 2), wh
  num_classes: integer

  Returns
  -------
  y_true: list of array, shape like yolo_outputs, xywh are reletive value

  '''
  assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
  num_layers = len(anchors)//3 # default setting
  anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

  true_boxes = np.array(true_boxes, dtype='float32')
  input_shape = np.array(input_shape, dtype='int32')
  boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
  boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
  true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
  true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

  m = true_boxes.shape[0]
  grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
  y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
      dtype='float32') for l in range(num_layers)]
  # Expand dim to apply broadcasting.
  anchors = np.expand_dims(anchors, 0)
  anchor_maxes = anchors / 2.
  anchor_mins = -anchor_maxes
  valid_mask = boxes_wh[..., 0]>0

  for b in range(m):
    # Discard zero rows.
    wh = boxes_wh[b, valid_mask[b]]
    if len(wh)==0: continue
    # Expand dim to apply broadcasting.
    wh = np.expand_dims(wh, -2)
    box_maxes = wh / 2.
    box_mins = -box_maxes

    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_area = wh[..., 0] * wh[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)

    # Find best anchor for each true box
    best_anchor = np.argmax(iou, axis=-1)

    for t, n in enumerate(best_anchor):
      for l in range(num_layers):
        if n in anchor_mask[l]:
          i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
          j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
          k = anchor_mask[l].index(n)
          c = true_boxes[b,t, 4].astype('int32')
          y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
          y_true[l][b, j, i, k, 4] = 1
          y_true[l][b, j, i, k, 5+c] = 1

  return y_true

def preprocess_img(
  image,
  boxes,
  input_shape = (416, 416),
  random = True,
  max_boxes = 20,
  jitter = 0.3,
  hue = 0.1,
  sat = 1.5,
  val = 1.5,
  proc_img = True
):
  iw, ih = image.shape[0], image.shape[1]
  h, w = input_shape
  box = boxes

  new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
  scale = rand(.25, 2)
  if new_ar < 1:
      nh = int(scale*h)
      nw = int(nh*new_ar)
  else:
      nw = int(scale*w)
      nh = int(nw/new_ar)

  image = tf.image.resize(image, (nw, nh), method=tf.image.ResizeMethod.BICUBIC)
  image = tf.keras.utils.array_to_img(image.numpy(), "channel_last")
  # place image
  dx = int(rand(0, w-nw))
  dy = int(rand(0, h-nh))
  new_image = Image.new('RGB', (w,h), (128,128,128))
  new_image.paste(image, (dx, dy))
  image = tf.keras.utils.img_to_array(new_image)

  # flip image or not
  flip = rand()<.5
  if flip: image = tf.image.flip_left_right(image)

  # distort image
  hue = rand(-hue, hue)
  sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
  val = rand(1, val) if rand()<.5 else 1/rand(1, val)
  x = tf.image.rgb_to_hsv(image/255.)
  x[..., 0] += hue
  x[..., 0][x[..., 0]>1] -= 1
  x[..., 0][x[..., 0]<0] += 1
  x[..., 1] *= sat
  x[..., 2] *= val
  x[x>1] = 1
  x[x<0] = 0
  image_data = tf.image.hsv_to_rgb(x) # numpy array, 0 to 1

  # correct boxes
  box_data = np.zeros((max_boxes,5))
  if len(box)>0:
    np.random.shuffle(box)
    box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
    box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
    if flip: box[:, [0,2]] = w - box[:, [2,0]]
    box[:, 0:2][box[:, 0:2]<0] = 0
    box[:, 2][box[:, 2]>w] = w
    box[:, 3][box[:, 3]>h] = h
    box_w = box[:, 2] - box[:, 0]
    box_h = box[:, 3] - box[:, 1]
    box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
    if len(box)>max_boxes: box = box[:max_boxes]
    box_data[:len(box)] = box

  return image_data, box_data

def coco_preprocessor(inputs):
  image = inputs["image"]
  bbox = inputs["objects"]["bbox"]
  image, boxes = preprocess_img(image, bbox, (416, 416))
  y_true = preprocess_true_boxes(boxes, (416, 416), yolo_tiny_anchors, 80)
  return (image, *y_true), np.zeros(8)



def data_generator(annon_lines, batch_size, input_shape, anchors, num_classes):
  def data_iter():
    n = len(annon_lines)
    i = 0
    while True:
      image_data = []
      box_data = []
      for _ in range(batch_size):
        # if i == 0:
        #   np.random.shuffle(annon_lines)
        image, box = get_random_data(annon_lines[i], input_shape, random=True)
        image_data.append(image)
        box_data.append(box)
        i = (i + 1) % n
      image_data = np.array(image_data)
      box_data = np.array(box_data)
      y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
      yield image_data, [*y_true, *y_true]
  return data_iter

sets = [('2012', 'train'), ('2012', 'val')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def convert_annotation(year, image_id, list_file):
  in_file = open('/u/home/nui/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
  tree=ET.parse(in_file)
  root = tree.getroot()
  for obj in root.iter('object'):
    difficult = obj.find('difficult').text
    cls = obj.find('name').text
    if cls not in classes or int(difficult)==1:
      continue
    cls_id = classes.index(cls)
    xmlbox = obj.find('bndbox')
    b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
    list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

def generate_voc_annotations():
  for year, image_set in sets:
    image_ids = open('/u/home/nui/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
      list_file.write('/u/home/nui/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(year, image_id))
      convert_annotation(year, image_id, list_file)
      list_file.write('\n')