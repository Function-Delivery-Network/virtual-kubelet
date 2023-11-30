"""Define default compile values for a model"""
from keras.optimizer_v2 import learning_rate_schedule
from keras.optimizer_v2.adam import Adam
def gen_default_compile_ops():
  lr = learning_rate_schedule.ExponentialDecay(
    initial_learning_rate=1.0e-2,
    decay_steps=6000,
    decay_rate=0.9,
  )

  optimizer = Adam(learning_rate=lr, decay=1e-5)
  return {
    "loss": "categorical_crossentropy",
    "optimizer": optimizer,
    "metrics": ["accuracy"]
  }