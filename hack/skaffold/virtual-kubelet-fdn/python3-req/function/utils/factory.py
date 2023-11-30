""" Early Exit function factory for multiple models"""
from keras import layers as keras_layers
from official.nlp.keras_nlp import layers
from official.nlp.modeling import layers as modeling_layers

from keras import activations
from keras import initializers

from ..backend.earlyexit import EarlyExit

def bert_early_exit_factory(
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
  use_encoder_pooler: bool = True,
  dropout_rate: float = 0.1,
  num_classes: int = 2,
  cls_initializer = "glorot_uniform",
  **kwargs
):
  """ Generates the BERT Classifier for GLUE/MRPC dataset """
  def layer_factory(inputs):
    inner_activation = lambda x: activations.gelu(x, approximate=True)
    initializer = initializers.TruncatedNormal(stddev=initializer_range)

    encoder = layers.TransformerEncoderBlock(
      num_attention_heads=num_attention_heads,
      inner_dim=intermediate_size,
      inner_activation=inner_activation,
      output_dropout=output_dropout,
      attention_dropout=attention_dropout_rate,
      norm_first=norm_first,
      output_range=None,
      kernel_initializer=initializer,
    )(inputs)

    data = keras_layers.Dense(
      units=hidden_size,
      activation="tanh",
      kernel_initializer=initializer,
    )(encoder[:,0,:])

    if use_encoder_pooler:
      cls_inputs = data
      cls_inputs = keras_layers.Dropout(rate=dropout_rate)(cls_inputs)
    else:
      cls_inputs = encoder
    
    classifier = modeling_layers.ClassificationHead(
      inner_dim=0,
      num_classes=num_classes,
      initializer=cls_initializer,
      dropout_rate=dropout_rate
    )(cls_inputs)

    return EarlyExit(inputs=[inputs[0], classifier])([inputs[0], classifier])
  return layer_factory
  
def yolo_tiny_early_exit_factory():
  """ Generates the YOLO Tiny Early Exit branch for VOC dataset """
  
