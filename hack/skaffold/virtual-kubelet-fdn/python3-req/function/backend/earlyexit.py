""" Description of the Early Exit layer """
from keras.engine.base_layer import Layer

class EarlyExit(Layer):
  """ Defines the Early Exit layer to be used by DistributedModel """
  def __init__(self, predict = False, inputs=None, **kwargs):
    """Creates the EarlyExit Layer for the Framework

    An EarlyExit layer is meant as an intermediate step for inference only.
    It will be inserted after training. The model will be deconstructed and
    an EarlyExit layer will be added between layers. The EarlyExit layer acts
    as a filter during inference.

    This layer must be trained only after the model has been trained.

    Args:
      **kwargs: Additional keyword arguments from tf.keras.layers.Layer.
        It defines features as `trainable`. The `name` of the layer, its
        `dtype`, and if it is `dynamic`.
    """
    self.predict = predict
    super().__init__(**kwargs)
    if inputs is not None:
      self.build(inputs)
  def get_config(self):
    config = super().get_config()
    config.update({'predict': self.predict})
    return config
  def build(self, input_shape):
    super().build(input_shape)
    self.built = True
  def compute_output_shape(self, input_shape):
    return input_shape
  def call(self, inputs, training=False):
    # if self.predict:
    #   raise EarlyExitException(inputs[-1])
    return inputs
