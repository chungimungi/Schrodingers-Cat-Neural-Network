# Schrodinger's Cat Neural Network Implementation

## Table of Contents
- [Introduction](#introduction)
- [Key Concepts](#key-concepts)
  - [Quantum Mechanics](#quantum-mechanics)
- [Code Structure](#code-structure)
  - [Quantum Randomness](#quantum-randomness)
  - [Weight Initialization](#weight-initialization)
  - [Probabilistic Activation Function](#probabilistic-activation-function)
  - [DeadOrAlive Loss Function (DOALoss)](#deadoralive-loss-function-doaloss)
  - [Model Architecture](#model-architecture)
  - [Training and Evaluation](#training-and-evaluation)
- [Data Preparation](#data-preparation)
- [Final Evaluation and Collapse](#final-evaluation-and-collapse)

## Introduction
The Schrodinger's Cat NN is a playful interpretation of the famous thought experiment in quantum mechanics, where a cat is both alive and dead until observed. This implementation uses two models (alive and dead) to represent the cat's states and incorporates probabilistic behaviors into the training process.

## Key Concepts

### Quantum Mechanics
- **Superposition**: In quantum mechanics, a system can exist in multiple states simultaneously. This is modeled in the NN by training two separate models that represent different states.
- **Collapse of the Wave Function**: Upon measurement, the superposition collapses into one of the possible states. In this implementation, this is simulated by evaluating the performance of both models and selecting one based on their accuracies.

## Code Structure

### Quantum Randomness
The `quantum_random` function generates a pseudo-random number using a sine function, simulating quantum randomness.

```python
def quantum_random():
    return np.sin(np.random.random() * np.pi) ** 2
```

### Weight Initialization
The init_weights function initializes the weights of the neural network layers using a uniform distribution and scales them by a quantum random value. This introduces variability akin to quantum behavior.
```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -np.sqrt(1/m.in_features), np.sqrt(1/m.in_features))
        m.weight.data = m.weight.data * quantum_random()
        m.bias.data.fill_(0.01)
```

### Probabilistic Activation Function
The ProbabilisticActivation class defines a custom activation function that outputs values based on the input's sign, ensuring outputs are within the range [0, π).
```python
class ProbabilisticActivation(nn.Module):
    def forward(self, x):
        positive_output = (x * torch.sin(x)**2) % np.pi  
        negative_output = (torch.rand_like(x) * x) % np.pi 
        return torch.where(x > 0, positive_output, negative_output)
```

### DeadOrAlive Loss Function (DOALoss)
Custom loss function named `DOALoss` is implemented. This loss function applies a weighted cross-entropy loss that emphasizes the importance of correctly classifying instances of the 'alive' and 'dead' classes based on their assigned weights.

### Model Architecture
The SchrodingersCatNN class defines a simple feedforward neural network with two linear layers and a custom activation function.

### Training and Evaluation
The SchrodingersCatBox class manages the training and evaluation of both models (alive and dead). It includes methods for training steps, evaluating performance, and collapsing the wave function based on the models' performances.
```python
class SchrodingersCatBox:
    def train_step(self, inputs, labels, criterion):
        # training logic for both models
        ...
    
    def evaluate(self, dataloader, criterion):
        # evaluation logic for both models
        ...
    
    def collapse_wave_function(self, performances):
        # logic to collapse the wave function based on performance
        ...
```

### Data Preparation
The dataset is generated using make_classification from sklearn, and it is split into training and testing sets. The data is then converted into PyTorch tensors and loaded into DataLoader for batch processing.

### Final Evaluation and Collapse
After training, the models are evaluated, and the wave function is collapsed based on their performances. The chosen model is then used for the final evaluation.
