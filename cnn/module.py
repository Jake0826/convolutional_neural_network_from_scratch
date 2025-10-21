import numpy as np 

class Module:
    """
    Base class for all neural network modules.
    
    All layers and loss functions inherit from this class,
    which provides a common interface for forward passes, backward passes,
    and gradient management.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        return dvalues
    
    def zero_grad(self) -> None:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)