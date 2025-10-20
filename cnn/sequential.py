import numpy as np 
from .layers import *
from .loss import * 
from .optim import * 
from .module import Module 
from typing import List

class Sequential(Module):
  def __init__(
    self, 
    layers: List[Module], 
    loss_func: Loss, 
    optimizer: Optimizer
  ):

    super().__init__()
    self.layers = layers
    self.loss_func = loss_func
    self.optimizer = optimizer

  def forward(self, x):
    for layer in self.layers:
      x = layer.forward(x)
    return x 
  
  def backward(self, y_pred: np.ndarray, y_hat: np.ndarray):
    dvalues = self.loss_func.backward(y_pred, y_hat)
    for layer in reversed(self.layers):
      dvalues = layer.backward(dvalues)
    return dvalues

  def optimize(self):
    for layer in self.layers:
      if hasattr(layer, 'weights'):
        self.optimizer.update_params(layer)
  
  def zero_grad(self):
    for layer in self.layers:
      if hasattr(layer, 'weights'): 
        layer.zero_grad() 

  def calculate_loss(self, output, target):
    return self.loss_func.calculate(output, target)