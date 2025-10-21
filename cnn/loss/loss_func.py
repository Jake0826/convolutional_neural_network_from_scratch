import numpy as np 
from ..module import Module

class Loss(Module):
  '''
  Parent Loss Function
  '''
  def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
    raise NotImplementedError("This method must be implemented by subclasses.")

  def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
    raise NotImplementedError("This method must be implemented by subclasses.")
  
  def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    samples_losses = self.forward(y_pred, y_true) 
    return np.mean(samples_losses)