import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import TensorDataset
from torch import Tensor
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.init as init
import torch.nn.functional as F
from typing import List, Tuple, Optional
from contextlib import nullcontext

'''
This file contains helper functions I used to train 
the reference PyTorch nerual networks.
'''
def pytorch_generate_dataloader(X, y, batch_size=32):
  X_tensor = torch.tensor(X, dtype=torch.float32)
  y_tensor = torch.tensor(y, dtype=torch.long)
  dataset = TensorDataset(X_tensor, y_tensor)
  dataloader = DataLoader(dataset, batch_size=batch_size)
  return dataloader


def pytorch_epoch_loop(
  model: torch.nn.Module, 
  dataloader: DataLoader,
  criterion: torch.nn.modules.loss._Loss,
  optimizer: Optional[torch.optim.Optimizer] = None,
  train: bool = True, 
) -> Tuple[float, float]:

  if train and optimizer is None:
    raise ValueError("Optimizer must be provided when train=True")
    
  model.train(train) 
  running_loss, correct, total = 0, 0, 0

  context = torch.no_grad() if not train else nullcontext()
  with context:
    for features, target in dataloader:  
      output = model.forward(features)
      loss = criterion(output, target)
      if train:
        optimizer.zero_grad() # type: ignore
        loss.backward()
        optimizer.step()  # type: ignore

      running_loss += loss.item() * features.shape[0]
      _, pred = torch.max(output, dim=1)
      correct += (pred == target).sum().item()
      total += target.shape[0]

  epoch_loss = running_loss / total
  epoch_accuracy = correct / total 

  return epoch_loss, epoch_accuracy

def pytorch_train_val(
  model: nn.Module, 
  dataloader_train: DataLoader,
  dataloader_val: DataLoader,
  criterion: torch.nn.modules.loss._Loss,
  optimizer: Optional[torch.optim.Optimizer],
  epochs: int = 100, 
  verbose_freq: int = 1
) -> Tuple[List[float], List[float], List[float], List[float]]:
  
  train_losses, train_accuracies = [], []
  val_losses, val_accuracies = [], []

  for epoch in range(epochs):
    train_loss, train_accuracy = pytorch_epoch_loop(
      model, 
      dataloader_train, 
      criterion,
      optimizer,
      train=True
    )
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    val_loss, val_accuracy = pytorch_epoch_loop(
      model, 
      dataloader_val,
      criterion,
      train=False
    )
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    if (epoch + 1) % verbose_freq == 0:
      print(f"Epoch {epoch + 1}, Train Loss: {train_loss:4f}, Train Accuracy: {train_accuracy:4f}, Val Loss: {val_loss:4f}, Val Accuracy: {val_accuracy:4f}")  

  return train_losses, train_accuracies, val_losses, val_accuracies

def pytorch_test(
  model: nn.Module, 
  dataloader: DataLoader, 
  criterion: torch.nn.modules.loss._Loss
) -> Tuple[float, float]:
  
  test_loss, test_accuracy = pytorch_epoch_loop(
    model,
    dataloader,
    criterion,
    train=False
  )
  print(f"Test Loss: {test_loss:4f}, Test Accuracy: {test_accuracy:4f}")
  return test_loss, test_accuracy  

def pytorch_plot_spiral_data_decision_boundary(model: nn.Module, title : str):
  x_1 = np.arange(-1, 1, 0.01)
  x_2 = np.arange(-1, 1, 0.01)
  X_1, X_2 = np.meshgrid(x_1, x_2)

  X_grid = np.c_[X_1.flatten(), X_2.flatten()]
  X_tensor = torch.tensor(X_grid, dtype=torch.float32)  

  model.eval()
  with torch.no_grad():
    output = model(X_tensor)

  _, pred = torch.max(output, dim=1)
  Z = pred.reshape(X_1.shape)

  plt.figure(figsize=(6, 5))
  plt.contourf(X_1, X_2, Z, cmap='brg')

  plt.xlim(-1,1)
  plt.ylim(-1,1)
  plt.xlabel('X_1')
  plt.ylabel('X_2')
  plt.title(title)
  plt.show()