import numpy as np

from initializers import random_initializer
from activations import relu
from typing import Callable

class DenseModel():
  def __init__(
      self,
      input_shape
      ):
    self.input_shape = input_shape
    self.trainable_params = 0

    self.layers: list = []
    self.weight_matrix = []
    self.layers_count = 0

  def add(
      self,
      layer
  ):
    if self.layers_count == 0:
      prev = self.input_shape
    else:
      prev = self.layers[-1].units
    curr = layer.units
    arr = np.random.rand(prev, curr)
    self.weight_matrix.append(arr)
    self.layers.append(layer)
    self.layers_count += 1

  def compile(
      self,
      learning_rate,
      optimizer,
      loss,
  ):
    pass

  def fit(
      self,
      X,
      y,
      epochs: int,
      batchsize: int
  ):
    pass

  def summary(self):
    pass