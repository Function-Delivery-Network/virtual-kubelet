""" Helper classes """
from enum import Enum

class ProfileMode(Enum):
  """ Defines the profile mode for the model """
  DISABLED    = 0
  FULL        = 1
  LAYER_ONLY  = 2
  DEPTH_ONLY  = 3

class ExecutionMode(Enum):
  """ Defines the execution mode for the model """
  PROFILE     = 0
  PARTIAL_RUN = 1
  NORMAL_EXEC = 2