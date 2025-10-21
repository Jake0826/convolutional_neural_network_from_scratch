# Convolutional Neural Network From Scratch

By building a fully functional Convolutional Neural Network from scratch, I now have a stronger theoretical foundation and deeper understanding of CNNs. I relied on NumPy for low-level matrix operations, as NumPy's C-based implementations are orders of magnitude faster than pure Python, but all the neural network logic is implemented from the ground up.

## Project Structure 

In the `cnn/` directory, I built out each component from scratch. In the `models/` directory, I provide two examples where I compare my custom implementation with PyTorch models using identical neural network architectures.

```
cnn/
├── layers/      
│   ├── conv2d.py
│   ├── maxpool2d.py
│   ├── linear.py
│   ├── relu.py
│   ├── softmax.py
│   ├── flatten.py
│   └── dropout.py
├── loss/           
│   ├── cross_entropy_loss.py
│   └── mse.py
│   └── loss_func.py
├── optim/ 
│   ├── optimizer.py
│   └── adam.py
├── utils/          
│   └── utils.py
├── module.py       
└── sequential.py    

models/             
├── spiral_dataset.ipynb
├── fashion_mnist.ipynb
└── pytorch_utils.ipynb  # Helper functions for PyTorch comparison models
```

## Usage Example
```python
from cnn.layers import Linear, ReLU, Softmax
from cnn.loss import CrossEntropyLoss
from cnn.optim import Adam
from cnn.sequential import Sequential
from cnn.utils import train_val

# Define model architecture
ordered_layers = [
    Linear(D, 32),
    ReLU(),
    Linear(32, 16),
    ReLU(),
    Linear(16, K),
    Softmax()
]

# Configure loss and optimizer
loss = CrossEntropyLoss()
optim = Adam(lr=1e-3)

# Build model
model = Sequential(
    layers=ordered_layers,
    loss_func=loss,
    optimizer=optim
)

# Train
EPOCHS = 100
batch_size = 32
train_losses, train_accuracies, val_losses, val_accuracies = train_val(
    model, X_train, y_train, X_val, y_val, 
    epochs=EPOCHS, 
    batch_size=batch_size, 
    verbose_freq=10
)
```

## Results

In the notebooks in the `models/` directory, I achieve nearly identical accuracy compared to PyTorch implementations on both the spiral dataset (NN) and Fashion-MNIST (CNN), validating that the custom implementation is mathematically correct.

## Installation
```bash
git clone https://github.com/Jake0826/convolutional_neural_network
cd convolutional_neural_network
pip install -e .
```

Check out the `models/` folder for complete examples and usage.

## Sources & Resources 

- [Neural Networks from Scratch (NNFS)](https://nnfs.io/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)



**Built to learn, not to scale** 🧠