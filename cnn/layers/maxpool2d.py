 
import numpy as np 
from ..module import Module

#Per CS231 notes 'For Pooling layers, it is not common to pad the input using zero-padding.'
class MaxPool2d(Module):
  def __init__(self, kernel_size=2, stride=2):
    self.kernel_size = kernel_size
    self.stride = stride 
  
# (B, C, H, W) -> (B, C, H_out, W_out)
  def forward(self, x):
      self.x = x
      
      pooled = self._get_strided_view(x)
      self.output = np.max(pooled, axis=(4, 5))
      
      B, C, H_out, W_out = self.output.shape
      pooled_flat = pooled.reshape(B, C, H_out, W_out, -1)
      self.max_indices = np.argmax(pooled_flat, axis=4)
      
      return self.output

  def backward(self, dvalues):
      dinputs = np.zeros_like(self.x)
      B, C, H_out, W_out = dvalues.shape
      
      max_h = self.max_indices // self.kernel_size
      max_w = self.max_indices % self.kernel_size
      
      b_idx = np.arange(B).reshape(-1, 1, 1, 1)
      c_idx = np.arange(C).reshape(1, -1, 1, 1)
      h_idx = np.arange(H_out).reshape(1, 1, -1, 1)
      w_idx = np.arange(W_out).reshape(1, 1, 1, -1)
      
      h_coords = h_idx * self.stride + max_h
      w_coords = w_idx * self.stride + max_w
      
      np.add.at(dinputs, (b_idx, c_idx, h_coords, w_coords), dvalues)
      
      return dinputs
  
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