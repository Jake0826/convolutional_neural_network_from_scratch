import numpy as np 
from ..module import Module

class Flatten(Module):
  def forward(self, x):
    self.x = x
    return x.reshape(x.shape[0], -1)
  
  def backward(self, dvalues):
    return dvalues.reshape(self.x.shape)
