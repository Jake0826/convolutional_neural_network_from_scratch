 
import numpy as np 
from ..module import Module

class Conv2d(Module):
  def __init__(self, in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride 
    self.padding = padding 

    self.weights = np.random.randn(
        self.out_channels, 
        self.in_channels, 
        self.kernel_size, 
        self.kernel_size
    ) * np.sqrt(2.0 / (self.in_channels * self.kernel_size * self.kernel_size))
    
    self.biases = np.zeros((self.out_channels))

  def forward(self, x):
      self.x = x
      self.x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
  
      self.x_strided = self._get_strided_view(self.x_padded)
      self.output = np.einsum('bchwkl,fckl->bfhw', self.x_strided, self.weights) + self.biases.reshape(1, -1, 1, 1)
      return self.output


  def backward(self, dvalues):
      self.dbiases = np.sum(dvalues, axis=(0, 2, 3))
      self.dweights = np.einsum('bfhw,bchwkl->fckl', dvalues, self.x_strided)
      
      dx_padded = np.zeros_like(self.x_padded)
      dx_strided = np.lib.stride_tricks.as_strided(
          dx_padded, 
          self.x_strided.shape, 
          self._get_strides(dx_padded), 
          writeable=True
      )
      grad_contribution = np.einsum('bfhw,fckl->bchwkl', dvalues, self.weights)
      dx_strided += grad_contribution
      
      if self.padding > 0:
          return dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
      return dx_padded

  def zero_grad(self):
    self.dweights = np.zeros_like(self.weights)
    self.dbiases = np.zeros_like(self.biases)

  def _get_strided_view(self, x):
      B, C, H, W = x.shape
      H_out = (H - self.kernel_size) // self.stride + 1
      W_out = (W - self.kernel_size) // self.stride + 1
      
      shape = (B, C, H_out, W_out, self.kernel_size, self.kernel_size)
      strides = self._get_strides(x)
      
      return np.lib.stride_tricks.as_strided(x, shape, strides)

  def _get_strides(self, x):
      return (
          x.strides[0],
          x.strides[1],
          x.strides[2] * self.stride,
          x.strides[3] * self.stride,
          x.strides[2],
          x.strides[3]
      )