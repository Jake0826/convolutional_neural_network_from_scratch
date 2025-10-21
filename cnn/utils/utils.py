import numpy as np 
from ..sequential import Sequential
from typing import Tuple, List
import matplotlib.pyplot as plt 

'''
This file contains helper functions I use in /model to train the neural network.
'''

# Dataset from https://cs231n.github.io/neural-networks-case-study/ 
def generate_spiral_data(N: int, D: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
  X = np.zeros((N*K,D)) # data matrix (each row = single example)
  y = np.zeros(N*K, dtype='uint8') # class labels
  for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
  # lets visualize the data:
  plt.xlabel("X_1")
  plt.ylabel("X_2")
  plt.title("Spiral Data")
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
  plt.show()

  return X, y

def plot_spiral_data_decision_boundary(model: Sequential, title: str = "") -> None:
  x_1 = np.arange(-1, 1, 0.01)
  x_2 = np.arange(-1, 1, 0.01)
  X_1, X_2 = np.meshgrid(x_1, x_2)

  X_grid = np.c_[X_1.flatten(), X_2.flatten()]
  output = model(X_grid)

  pred = np.argmax(output, axis=1)
  Z = pred.reshape(X_1.shape)

  plt.figure(figsize=(6, 5))
  plt.contourf(X_1, X_2, Z, cmap='brg')

  plt.xlim(-1,1)
  plt.ylim(-1,1)
  plt.xlabel('X_1')
  plt.ylabel('X_2')
  plt.title(title)
  plt.show()

def train_val_test_split(
  X: np.ndarray, 
  y: np.ndarray, 
  train_pct: float = 0.7, 
  val_pct: float = 0.15, 
  test_pct: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  

  assert np.isclose(train_pct + val_pct + test_pct, 1.0), "Ratios must sum to 1"

  size = len(X)
  indices = np.random.permutation(size)
  X = X[indices]
  y = y[indices]

  train_end = int(train_pct * size)
  val_end = train_end + int(val_pct * size)

  X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
  y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

  return X_train, X_val, X_test, y_train, y_val, y_test

def batch_generator(
    X: np.ndarray, 
    y: np.ndarray, 
    batch_size: int = 32
): 
  num_samples = X.shape[0]
  indicies = np.arange(num_samples)
  for i in range(0, num_samples, batch_size):
    j = min(i + batch_size, num_samples)
    idx = indicies[i:j]
    yield X[idx], y[idx]


def epoch_loop(
  model: Sequential, 
  X: np.ndarray, 
  y: np.ndarray, 
  train: bool = True, 
  batch_size: int = 32
) -> Tuple[float, float]:
  
  running_loss, correct, total = 0, 0, 0
  for idx, (features, target) in enumerate(batch_generator(X, y, batch_size)):
    
    target = target.flatten()

    model.zero_grad()
    
    output = model.forward(features)
    loss = model.calculate_loss(output, target)

    if train:
      model.backward(output, target)
      model.optimize()

    running_loss += loss * features.shape[0]
    pred = output.argmax(axis=1)
    correct += (pred == target).sum().item()
    total += target.shape[0]

  epoch_loss = running_loss / X.shape[0]
  epoch_accuracy = correct / total 

  return epoch_loss, epoch_accuracy

def train_val(
  model: Sequential, 
  X_train: np.ndarray,
  y_train: np.ndarray, 
  X_val: np.ndarray, 
  y_val: np.ndarray,
  epochs: int = 100, 
  batch_size: int = 32,
  verbose_freq: int = 1
) -> Tuple[List, List, List, List]:
  
  train_losses, train_accuracies = [], []
  val_losses, val_accuracies = [], []

  for epoch in range(epochs):
    train_loss, train_accuracy = epoch_loop(model, X_train, y_train, train=True, batch_size=batch_size)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    val_loss, val_accuracy = epoch_loop(model, X_val, y_val, train=False, batch_size=batch_size)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    if (epoch + 1) % verbose_freq == 0:
      print(f"Epoch {epoch + 1}, Train Loss: {train_loss:4f}, Train Accuracy: {train_accuracy:4f}, Val Loss: {val_loss:4f}, Val Accuracy: {val_accuracy:4f}")  

  return train_losses, train_accuracies, val_losses, val_accuracies

def plot_training_metrics(
  train_losses: List[float], 
  train_accuracies: List[float], 
  val_losses: List[float], 
  val_accuracies: List[float], 
  title: str = "",
) -> None:
  
  epochs = np.arange(1, len(train_losses) + 1)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  
  ax1.plot(epochs, train_losses, label='Train', linewidth=2, marker='o', markersize=3, alpha=0.8)
  ax1.plot(epochs, val_losses, label='Validation', linewidth=2, marker='s', markersize=3, alpha=0.8)
  ax1.set_title(f"{title} Cross Entropy Loss", fontweight='bold')
  ax1.set_xlabel('Epoch', fontsize=11)
  ax1.set_ylabel('Loss', fontsize=11)
  ax1.legend()
  ax1.set_ylim(ymin = 0)
  
  ax2.plot(epochs, train_accuracies, label='Train', linewidth=2, marker='o', markersize=3, alpha=0.8)
  ax2.plot(epochs, val_accuracies, label='Validation', linewidth=2, marker='s', markersize=3, alpha=0.8)
  ax2.set_title(f"{title} Classification Accuracy", fontweight='bold')
  ax2.set_xlabel('Epoch', fontsize=11)
  ax2.set_ylabel('Accuracy', fontsize=11)
  ax2.legend()
  ax2.set_ylim(ymin = 0, ymax = 1)

  plt.tight_layout()
  plt.show()


def test(
  model: Sequential, 
  X_test: np.ndarray, 
  y_test: np.ndarray
) -> Tuple[float, float]:
  
  test_loss, test_accuracy = epoch_loop(model, X_test, y_test, train=False)
  print(f"Test Loss: {test_loss:4f}, Test Accuracy: {test_accuracy:4f}")
  return test_loss, test_accuracy  