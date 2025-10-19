import numpy as np 
from ..module import Module

class ReLU(Module):
  def forward(self, x):
    self.x = x 
    return np.maximum(0, self.x)
  
  def backward(self, dvalues):
    return dvalues * np.where(self.x > 0, 1, 0)