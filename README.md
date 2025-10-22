# Convolutional Neural Network From Scratch

## Project Overview

I built a fully functional Convolutional Neural Network from scratch to better understand neural networks.  The `/cnn` directory contains all the building blocks required to construct and train a CNN from the ground up. The `/models` directory implements the neural networks by training on two seperate datasets and benchmarking performance against equivalent PyTorch models.

## Architecture

The `/cnn` directory contains the building blocks, with each component in its own Python file. All layers inherit from a base `Module` class that includes forward and backward methods. The `Sequential` class, which accepts an ordered list of layers, loss function, and optimizer, is the conductor that chains the pieces together.

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
│   ├── mse.py
│   └── loss_func.py
├── optim/  
│   ├── optimizer.py
│   └── adam.py
├── utils/           
│   └── utils.py
├── module.py        
└── sequential.py     
```

## Usage Example
```python
from cnn.layers import Linear, ReLU, Softmax, Conv2d, MaxPool2d, Flatten
from cnn.loss import CrossEntropyLoss
from cnn.optim import Adam
from cnn.sequential import Sequential
from cnn.utils import train_val

# Define architecture
model = Sequential(
    layers=[

    Conv2d(in_channels=1, out_channels=16, kernel_size=3),

    ReLU(),

    MaxPool2d(kernel_size = 2, stride = 2),

    Flatten(),

    Linear(1024, num_classes),

    Softmax()

    ],

    loss_func=CrossEntropyLoss(),
    optimizer=Adam(lr=1e-3)
)

# Train
train_losses, train_accuracies, val_losses, val_accuracies = train_val(
    model, X_train, y_train, X_val, y_val, 
    epochs=100, 
    batch_size=32, 
    verbose_freq=10
)
```


## Results

The `/models` directory contains experiments comparing my implementation against PyTorch models of identical architecture on two separate datasets. All training was performed locally using my Mac M1 CPU.

## Spiral Dataset 

![Spiral Dataset](images/spiral/data.png)

In `/models/spiral.ipynb`, I trained a fully connected neural network on a 2D coordinate classification task with 8 classes. The dataset was borrowed from Stanford CS231n's course website. The model uses 3 linear layers with ReLU activations, trained with the Adam optimizer (lr=1e-3) and cross-entropy loss.
```python
ordered_layers = [
    Linear(2, 32),
    ReLU(),      
    Linear(32, 16),
    ReLU(),
    Linear(16, 8),
    Softmax()
]
```

| Metric | Homemade Implementation | PyTorch Implementation |
|--------|----------------------|------------------------|
| Test Loss | 0.368 | 0.386 |
| Test Accuracy | 86.3% | 85.0% |
| Training Time | 5.2s | 14.6s |

### Homemade Implementation
![Homemade Model Metrics](images/spiral/homemade_metrics.png)
![Homemade Decision Boundaries](images/spiral/homemade_decision_boundaries.png)

### PyTorch Implementation
![PyTorch Metrics](images/spiral/pytorch_metrics.png)
![PyTorch Decision Boundaries](images/spiral/pytorch_decision_boundaries.png)

## Fashion MNIST 

![Fashion MNIST Dataset](images/fashion_mnist/data.png)

In `/models/fashion_mnist.ipynb`, I trained a convolutional neural network to classify 10 clothing items from the Fashion MNIST dataset. The model architecture is:
```python
ordered_layers = [
    Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    
    Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),

    Flatten(),

    Linear(1568, 128),
    ReLU(),
    Dropout(),
    
    Linear(128, 10),
    Softmax()
]
```

| Metric | Homemade Implementation | PyTorch Implementation |
|--------|----------------------|------------------------|
| Test Loss | 0.310 | 0.223 |
| Test Accuracy | 89.3% | 91.6% |
| Training Time | 80m | 3m |

### Homemade Implementation
![Homemade Model Metrics](images/fashion_mnist/homemade_metrics.png)

### PyTorch Implementation
![PyTorch Metrics](images/fashion_mnist/pytorch_metrics.png)

## Conclusion 

I now have a much deeper understanding of CNNs. It's incredibly rewarding to have built a model from scratch that works correctly—even if its performance doesn't match PyTorch's highly optimized implementations, particularly with `Conv2d` and `MaxPool2d` layers, which are harder to optimize in Python.

While CNNs are no longer the cutting edge of deep learning, and I may never resort to a CNN model, I think it's essential for me to understand the foundational ideas that led to today's leading architectures. My next project will focus on RNNs and language modeling before moving to attention. 

## tldr 

Up to now, I've mostly been a practitioner—but my goal over the coming months is to move beyond implementation and truly understand the underlying principles behind AI models.

## References 

- [Neural Networks from Scratch](https://nnfs.io/) 
- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.stanford.edu/) 
- [Vizuara - Building Neural Networks from Scratch](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSj6tNyn_UadmUeU3Q3oR-hu) 
- [Andrej Karpathy - Autograd](https://www.youtube.com/watch?v=VMj-3S1tku0) 