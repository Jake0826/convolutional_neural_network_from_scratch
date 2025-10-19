import numpy as np 
from ..module import Module

class Loss(Module):
  def calculate(self, output, y):
    samples_losses = self.forward(output, y) # type: ignore
    return np.mean(samples_losses)