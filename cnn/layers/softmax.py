import numpy as np 
from ..module import Module

class Softmax(Module):
  def forward(self, x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return self.output

  def backward(self, dvalues):
    self.dx = np.empty_like(dvalues)
    for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
      single_output = single_output.reshape(-1, 1)
      jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
      self.dx[index] = np.dot(jacobian_matrix, single_dvalues)
    return self.dx