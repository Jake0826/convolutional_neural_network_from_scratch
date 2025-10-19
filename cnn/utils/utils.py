import numpy as np 

def train_val_test_split(X, y, train_pct=0.7, val_pct=0.15, test_pct=0.15):
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

def batch_generator(X, y, batch_size=32):
  num_samples = X.shape[0]
  indicies = np.arange(num_samples)
  for i in range(0, num_samples, batch_size):
    j = min(i + batch_size, num_samples)
    idx = indicies[i:j]
    yield X[idx], y[idx]


def epoch_loop(model, X, y, train=True, batch_size=32):
  running_loss, correct, total = 0, 0, 0
  for features, target in batch_generator(X, y, batch_size):
    output = model.forward(features)
    loss = model.calculate_loss(output, target)

    if train:
      model.backward(output, target)
      model.optimize()
      model.zero_grad()

    running_loss += loss * features.shape[0]
    pred = output.argmax(axis=1)#.reshape(target.shape[0], 1)
    correct += (pred == target).sum().item()
    total += target.shape[0]

  epoch_loss = running_loss / X.shape[0]
  epoch_accuracy = correct / total # type: ignore

  if train:
    model.optimizer.post_update_params()

  return epoch_loss, epoch_accuracy

def train(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
  train_losses, train_accuracies = [], []
  val_losses, val_accuracies = [], []

  for epoch in range(epochs):
    train_loss, train_accuracy = epoch_loop(model, X_train, y_train, train=True)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    val_loss, val_accuracy = epoch_loop(model, X_val, y_val, train=False)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:4f}, Train Accuracy: {train_accuracy:4f}, Val Loss: {val_loss:4f}, Val Accuracy: {val_accuracy:4f}")  

  return train_losses, train_accuracies, val_losses, val_accuracies