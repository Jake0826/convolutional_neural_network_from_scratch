import numpy as np 
from ..module import Module
from .loss_func import Loss 

class MSE(Loss):
  def forward(self, y_pred, y_true):
    self.output = np.power((y_pred - y_true),2)
    return self.output
  
  def backward(self, y_pred, y_true):
    batch_size = len(y_pred)
    self.dx = 2 * (y_pred - y_true)
    self.dx /= batch_size
    return self.dx