class Module:

  def forward(self, x):
    return x
  
  def backward(self, dvalues):
    return dvalues 
  
  def zero_grad(self):
    pass

  def __call__(self, x):
    return self.forward(x)

