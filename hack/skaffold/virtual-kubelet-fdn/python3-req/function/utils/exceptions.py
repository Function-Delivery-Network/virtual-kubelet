""" User-defined exceptions """
class EarlyExitException(BaseException):
  """ Exception raised when an EarlyExit layer is encountered during inference

      It avoids the exception handler from TF.
  """
  def __init__(self, value):
    self.value = value
    pass