import numpy as np 
from ..module import Module

class Linear(Module):
  def __init__(self, n_inputs: int, n_neurons: int):
    k = np.sqrt(2 / n_inputs)
    shape = (n_inputs, n_neurons)
    self.weights = np.random.uniform(-k, k, shape)
    self.biases =  np.zeros((1, n_neurons))

  def forward(self, x):
    self.x = x 
    return np.dot(self.x, self.weights) + self.biases 
  
  def backward(self, dvalues):
    self.dweights = np.dot(self.x.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    return np.dot(dvalues, self.weights.T)

  def zero_grad(self):
    self.dweights = np.zeros_like(self.weights)
    self.dbiases = np.zeros_like(self.biases)
