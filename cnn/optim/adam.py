
import numpy as np 
from .optimzer import Optimizer
from ..module import Module
class Adam(Optimizer):
  def __init__(
      self, 
      lr: float = 0.01, 
      decay: float = 0.0, 
      eps: float = 1e-7, 
      beta_1: float = 0.9, 
      beta_2: float = 0.999
    ):
    
    self.lr_0 = lr 
    self.lr = lr 
    self.decay = decay
    self.iterations = 0 
    self.eps = eps 
    self.beta_1 = beta_1
    self.beta_2 = beta_2
  
  def initialize(self, layer):
    layer.weights_momentums = np.zeros_like(layer.weights)
    layer.weights_cache = np.zeros_like(layer.weights)
    layer.biases_momentums = np.zeros_like(layer.biases)
    layer.biases_cache = np.zeros_like(layer.biases)

  def update_params(self, layer):
    if not hasattr(layer, 'weights_momentums'):
      self.initialize(layer)
    self.lr = self.lr_0 * (1 / (1 + self.decay * self.iterations))

    layer.weights_momentums = self.beta_1 * layer.weights_momentums + (1 - self.beta_1) * layer.dweights
    layer.biases_momentums = self.beta_1 * layer.biases_momentums + (1 - self.beta_1) * layer.dbiases

    weights_momentums_corrected = layer.weights_momentums / (1 - self.beta_1 ** (self.iterations + 1))
    biases_momentums_corrected = layer.biases_momentums / (1 - self.beta_1 ** (self.iterations + 1))

    layer.weights_cache = self.beta_2 * layer.weights_cache + (1 - self.beta_2) * layer.dweights ** 2
    layer.biases_cache = self.beta_2 * layer.biases_cache + (1 - self.beta_2) * layer.dbiases ** 2

    weights_cache_corrected = layer.weights_cache / (1 - self.beta_2 ** (self.iterations + 1))
    biases_cache_corrected = layer.biases_cache / (1 - self.beta_2 ** (self.iterations + 1))
    
    layer.weights = layer.weights - self.lr * weights_momentums_corrected / np.sqrt(weights_cache_corrected + self.eps)
    layer.biases = layer.biases - self.lr * biases_momentums_corrected / np.sqrt(biases_cache_corrected + self.eps)

  def post_update_params(self):
    self.iterations += 1
