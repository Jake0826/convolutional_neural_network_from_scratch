import numpy as np 
from ..module import Module

class Dropout(Module):
  def __init__(self, p: float = 0.5):
    self.rate = 1 - p 

  def forward(self, x):
    self.x = x
    self.binary_mask = np.random.binomial(1, self.rate, size=x.shape) / (self.rate)
    return x * self.binary_mask 
  
  def backward(self, dvalues):
    self.dx = dvalues * self.binary_mask 
    return self.dx
