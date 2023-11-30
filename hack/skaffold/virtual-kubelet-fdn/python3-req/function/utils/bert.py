""" Specification for the BERT model for testing """
import os
import json
import random
import numpy as np

from keras.optimizer_v2 import learning_rate_schedule, adam
from keras.losses import SparseCategoricalCrossentropy
from keras import optimizers
from official.nlp.keras_nlp import layers
from official.nlp.modeling import layers as model_layers
import tensorflow as tf
from official.nlp.modeling.layers import BertTokenizer, BertPackInputs
from official.nlp.configs.encoders import EncoderConfig, build_encoder
from official.nlp.modeling.models.bert_classifier import BertClassifier
from official.modeling.optimization.lr_schedule import LinearWarmup
from official.modeling import tf_utils
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from ..backend.model import DistributedModel
from ..backend.earlyexit import EarlyExit
from ..utils.factory import bert_early_exit_factory

import matplotlib.pyplot as plt

class BertInputProcessor(tf.keras.layers.Layer):
  def __init__(self, tokenizer, packer):
    super().__init__()
    self.tokenizer = tokenizer
    self.packer = packer
  def call(self, inputs):
    tok1 = self.tokenizer(inputs["sentence1"])
    tok2 = self.tokenizer(inputs["sentence2"])
    packed = self.packer([tok1, tok2])
    if 'label' in inputs:
      return packed, inputs['label']
    else:
      return packed

def generate_bert_encoder(
  vocab_size: int = 30522,
  hidden_size: int = 768,
  num_layers: int = 12,
  num_attention_heads: int = 12,
  intermediate_size: int = 3072,
  hidden_activation: str = "gelu",
  output_dropout = 0.1,
  attention_dropout_rate: float = 0.1,
  max_sequence_length: int = 512,
  type_vocab_size: int = 2,
  initializer_range: float = 0.02,
  output_range = None,
  embedding_width = None,
  embedding_layer = None,
  return_all_encoder_outputs: bool = False,
  dict_outputs: bool = True,
  norm_first: bool = False,
  **kwargs
):
  """ Generates the BERT Encoder structure """
  activation = tf_utils.get_activation(hidden_activation)
  initializer = tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

  word_ids = tf.keras.layers.Input(
    shape=(None,), dtype=tf.int32, name="input_word_ids"
  )
  mask = tf.keras.layers.Input(
    shape=(None,), dtype=tf.int32, name="input_mask"
  )
  type_ids = tf.keras.layers.Input(
    shape=(None,), dtype=tf.int32, name="input_type_ids"
  )

  if embedding_width is None:
    embedding_width = hidden_size
  
  if embedding_layer is None:
    embedding_layer = layers.OnDeviceEmbedding(
      vocab_size=vocab_size,
      embedding_width=embedding_width,
      initializer=initializer,
      name="word_embeddings",
    )
  word_embeddings = embedding_layer(word_ids)

  pos_embedding_layer = layers.PositionEmbedding(
    initializer=initializer,
    max_length=max_sequence_length,
    name="position_embedding",
  )

  pos_embedding = pos_embedding_layer(word_embeddings)

  type_embedding_layer = layers.OnDeviceEmbedding(
    vocab_size=type_vocab_size,
    embedding_width=embedding_width,
    initializer=initializer,
    use_one_hot=True,
    name="type_embeddings",
  )
  type_embeddings = type_embedding_layer(type_ids)

  embeddings = tf.keras.layers.Add()([word_embeddings, pos_embedding, type_embeddings])

  embedding_norm_layer = tf.keras.layers.LayerNormalization(
    name="embeddings/layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32
  )

  embeddings = embedding_norm_layer(embeddings)
  embeddings = (tf.keras.layers.Dropout(rate=output_dropout)(embeddings))

  if embedding_width != hidden_size:
    embedding_projection = tf.keras.layers.experimental.EinsumDense(
        '...x,xy->...y',
        output_shape=hidden_size,
        bias_axes='y',
        kernel_initializer=initializer,
        name='embedding_projection')
    embeddings = embedding_projection(embeddings)
  else:
    embedding_projection = None
  
  transformer_layers = []
  early_exit = []
  data = embeddings
  attention_mask = layers.SelfAttentionMask()(data, mask)
  encoder_outputs = []
  inner_activation = lambda x: tf.keras.activations.gelu(x, approximate=True)
  ee_idx = [True, False, True, False, True, False, True, False, True, False, True, False]
  for i in range(num_layers):
    if i == num_layers -1 and output_range is not None:
      transformer_output_range = output_range
    else:
      transformer_output_range = None
    layer = layers.TransformerEncoderBlock(
      num_attention_heads=num_attention_heads,
      inner_dim=intermediate_size,
      inner_activation=inner_activation,
      output_dropout=output_dropout,
      attention_dropout=attention_dropout_rate,
      norm_first=norm_first,
      output_range=transformer_output_range,
      kernel_initializer=initializer,
      name="transformer/layer_%d" % i,
    )
    transformer_layers.append(layer)
    data = layer([data, attention_mask])
    # if ee_idx[i]:
    #   exitblock_cls = bert_early_exit_factory(data, attention_mask,
    #     num_attention_heads, intermediate_size, inner_activation,
    #     output_dropout, attention_dropout_rate, norm_first,
    #     transformer_output_range, initializer, hidden_size, 0.1,
    #     2
    #   )
    #   data, exits = EarlyExit(inputs=[data, exitblock_cls])([data, exitblock_cls])
    #   early_exit.append(exits)
    encoder_outputs.append(data)
  
  last_encoder_output = encoder_outputs[-1]
  first_token_tensor = last_encoder_output[:, 0, :]
  pooler_layer = tf.keras.layers.Dense(
    units=hidden_size,
    activation="tanh",
    kernel_initializer=initializer,
    name="pooler_transform",
  )
  cls_output = pooler_layer(first_token_tensor)
  outputs = dict(
    sequence_output=encoder_outputs[-1],
    pooled_output=cls_output,
    encoder_outputs=encoder_outputs,
    # early_exit=early_exit
  )
  return word_ids, mask, type_ids, outputs

def generate_bert_classifier(
  word_ids,
  mask,
  type_ids,
  outputs,
  num_classes,
  initializer="glorot_uniform",
  dropout_rate=0.1,
  use_encoder_pooler=True,
  head_name="sentence_prediction",
  cls_head=None
  ):
  """ Generates the BERT Classifier for GLUE/MRPC dataset """
  inputs = [word_ids, mask, type_ids]
  if use_encoder_pooler:
    if isinstance(outputs, list):
      cls_inputs = outputs[1]
    else:
      cls_inputs = outputs["pooled_output"]
    cls_inputs = tf.keras.layers.Dropout(rate=dropout_rate)(cls_inputs)
  else:
    if isinstance(outputs, list):
      cls_inputs = outputs[1]
    else:
      cls_inputs = outputs["sequence_output"]
  if cls_head:
    classifier = cls_head
  else:
    classifier = model_layers.ClassificationHead(
      inner_dim = 0 if use_encoder_pooler else cls_inputs.shape[-1],
      num_classes=num_classes,
      initializer=initializer,
      dropout_rate=dropout_rate,
      name=head_name,
    )
  predictions = classifier(cls_inputs)
  return predictions

def train_bert_model(model, epochs=10):
  """ Trains the BERT model for GLUE/MRPC dataset """
  batch_size = 32
  gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
  max_seq_length = 128
  glue, info = tfds.load("glue/mrpc", with_info=True, batch_size=batch_size)

  tokenizer = BertTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
    lower_case=True,
  )

  packer = BertPackInputs(
    seq_length=max_seq_length,
    special_tokens_dict=tokenizer.get_special_tokens_dict(),
  )

  bert_preprocess = BertInputProcessor(tokenizer, packer)

  glue_train = glue["train"].map(bert_preprocess).prefetch(1)
  glue_validation = glue["validation"].map(bert_preprocess).prefetch(1)
  # glue_test = glue["test"].map(bert_preprocess).prefetch(1)

  train_data_size = info.splits["train"].num_examples
  steps_per_epoch = int(train_data_size / batch_size)
  num_train_steps = steps_per_epoch * epochs
  warmup_steps = int(0.1*num_train_steps)
  initial_lr = 3e-5

  linear_decay = learning_rate_schedule.PolynomialDecay(
    initial_learning_rate=initial_lr,
    end_learning_rate=0,
    decay_steps=num_train_steps,
  )

  warmup_schedule = LinearWarmup(
    warmup_learning_rate=0,
    after_warmup_lr_sched=linear_decay,
    warmup_steps=warmup_steps,
  )
  optimizer = adam.Adam(
    learning_rate=warmup_schedule,
  )

  loss = SparseCategoricalCrossentropy(from_logits=True)
  
  

  model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=["accuracy"],
  )
  model.evaluate(glue_validation)

  model.fit(
    glue_train,
    validation_data=glue_validation,
    batch_size=32,
    epochs=epochs,
  )

def predict_bert_model(model):
  batch_size = 32
  gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
  max_seq_length = 128
  glue, info = tfds.load("glue/mrpc", with_info=True, batch_size=1)

  tokenizer = BertTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
    lower_case=True,
  )

  packer = BertPackInputs(
    seq_length=max_seq_length,
    special_tokens_dict=tokenizer.get_special_tokens_dict(),
  )

  bert_preprocess = BertInputProcessor(tokenizer, packer)

  glue_train = glue["train"].map(bert_preprocess).prefetch(1)
  glue_validation = glue["validation"].map(bert_preprocess).prefetch(1)
  def eval_bert(thr = 2):
    def eval_fn(outputs, thr = 0.5):
      max_by_row = tf.math.reduce_max(outputs, axis=1)
      return tf.reduce_mean(max_by_row) > thr
    return eval_fn
  model.evaluator = eval_bert()
  outputs = model.predict(glue_train, batch_size=batch_size, verbose=1)
  glue_train = iter(glue_train)
  acc = [0,0,0,0,0,0,0]
  
  
  for idx, (sample, label) in enumerate(glue_train):
    for ido, output in enumerate(outputs):
      if label[0] == tf.argmax(output[idx]):
        acc[ido] += 1

  return np.asarray(acc)/len(outputs[0])
  
def generate_bert_model(config):
  """ Generates the BERT model for GLUE/MRPC dataset """

  gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
  word_ids, mask, type_ids, outputs = generate_bert_encoder(**config)
  # encoder = DistributedModel([word_ids, mask, type_ids], outputs)
  # print("Loading weights from checkpoint")
  # checkpoint = tf.train.Checkpoint(encoder=encoder)
  # checkpoint.read(
  #   "/u/home/nui/saved_model.pb"
  # ).assert_consumed()
  predictions = generate_bert_classifier(word_ids, mask, type_ids, outputs, 2)
  

  model =  DistributedModel(
    inputs=[word_ids, mask, type_ids],
    # outputs=[predictions, *outputs["early_exit"]],
    outputs=predictions,
  )
  # gg = tf.keras.models.load_model("/u/home/nui/saved_model")
  # model.weights = gg.weights
  return model