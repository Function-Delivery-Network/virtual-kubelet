""" Module to provide a context manager for timing code execution. """
import time

class ExeTimer(object):
  """ Defines a context manager for timing code execution """
  def __init__(self, name=None):
    self.name = name
    self.visited_layers = list()
    self.execution = 0
    self.tstart = 0
  def __enter__(self, model = None):
    self.tstart = time.time()
    return self
  def current(self):
    """ Returns the current execution time """
    return time.time()-self.tstart
  def __exit__(self, type, value, traceback):
    self.execution = time.time()-self.tstart
