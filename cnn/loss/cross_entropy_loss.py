import numpy as np 
from ..module import Module
from .loss_func import Loss 

class CrossEntropyLoss(Loss):
  def forward(self, y_pred, y_true):
    y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
    correct_confidences = y_pred_clipped[range(len(y_pred)), y_true]
    return -np.log(correct_confidences)
  
  def backward(self, dvalues, y_true):
    batch_size = len(dvalues)
    unique_labels = len(dvalues[0])
    dvalues = np.clip(dvalues, 1e-7, 1-1e-7)
    y_true = np.eye(unique_labels)[y_true]
    self.dx = -y_true / dvalues
    self.dx /= batch_size
    return self.dx
