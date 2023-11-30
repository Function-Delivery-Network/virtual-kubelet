from typing import Optional


from pydantic import BaseModel

class RequestModel(BaseModel):
  forcefull: bool = False
  inputbucket: str = "inputs"
  inputname: str = "input.pkl"
  modelstore: str = "saved-models"
  modelname: str = "resnet50"
  dataset: str = "cifar10"
  modelpart: int = 0 # 0: head, 1: tail
  earlyexit: bool = True
  miniohost:       str = "138.246.237.238:30010"
  minioaccess:     str = "minio"
  miniopass:       str = "0Wo7gXXY2PWkB4i"
  batchsize:       int = 1
  tailendpoint:   str = "http://138.246.237.238:31112/function/requesthandler/tail"
  threshold:       float = 0.5
  outputnames:     Optional[str] = None
  output:         Optional[list] = None
  class Config:
    """ Pydantic configuration """
    arbitrary_types_allowed = True
