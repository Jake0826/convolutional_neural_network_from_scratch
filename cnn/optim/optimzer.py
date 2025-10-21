class Optimizer:
  '''
  Parent Optimizer Class
  '''
  def __init__(self, lr=0.01):
    self.lr = lr

  def update_params(self, layer):
    layer.weights += -self.lr * layer.dweights
    layer.biases += -self.lr * layer.dbiases
    