import numpy as np 
from ..module import Module

class Dropout(Module):
  def __init__(self, p: float = 0.5):
    self.keep_prob = 1 - p 

  def forward(self, x, train = True):
    if train:
      self.x = x
      self.binary_mask = np.random.binomial(1, self.keep_prob, size=x.shape) / (self.keep_prob)
      return x * self.binary_mask 
    return x 
  
  def backward(self, dvalues):
    self.dx = dvalues * self.binary_mask 
    return self.dx
