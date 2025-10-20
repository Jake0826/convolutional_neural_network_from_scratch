import numpy as np 

class Module:

  def forward(self, x: np.ndarray) -> np.ndarray:
    return x
  
  def backward(self, dvalues: np.ndarray) -> np.ndarray:
    return dvalues 
  
  def zero_grad(self):
    pass

  def __call__(self, x: np.ndarray) -> np.ndarray:
    return self.forward(x)

