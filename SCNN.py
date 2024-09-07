import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DOALoss(nn.Module):
    """
    Custom loss function for modeling the behavior of a system where 
    the importance of predictions differs for 'alive' and 'dead' classes.

    This loss function applies weighted cross-entropy loss, allowing 
    for greater emphasis on correctly predicting instances of either 
    class based on their assigned weights.

    Attributes:
        alive_weight (float): Weight assigned to the 'alive' class. 
                              Defaults to 1.0.
        dead_weight (float): Weight assigned to the 'dead' class. 
                             Defaults to 1.0.

    Methods:
        forward(outputs, labels):
            Computes the weighted cross-entropy loss based on the 
            provided outputs and labels.

    Args:
        alive_weight (float): The weight to apply for the 'alive' class.
        dead_weight (float): The weight to apply for the 'dead' class.
    """

    def __init__(self, alive_weight=1.0, dead_weight=1.0):
        super(DOALoss, self).__init__()
        self.alive_weight = alive_weight
        self.dead_weight = dead_weight

    def forward(self, outputs, labels):
        loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
        weights = torch.where(labels == 0, self.alive_weight, self.dead_weight)
        return (loss * weights).mean()


def quantum_random():
    """
    Generates a quantum-inspired random number between 0 and 1.

    Uses the sine function to create a non-uniform distribution, simulating quantum randomness.

    Returns:
        float: A random number between 0 and 1.
    """
    return np.sin(np.random.random() * np.pi) ** 2

def init_weights(m):
    """
    Initializes weights of linear layers using a uniform distribution.

    Applies quantum randomness to the weights and sets the bias to a small positive value.

    Args:
        m (nn.Module): The neural network module to initialize.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -np.sqrt(1/m.in_features), np.sqrt(1/m.in_features))
        m.weight.data *= quantum_random()
        m.bias.data.fill_(0.01)

class ProbabilisticActivation(nn.Module):
    """
    Custom activation function that introduces probabilistic behavior.

    Transforms input values to the range [0, pi) using different methods for positive and negative inputs.
    """
    def forward(self, x):
        positive_output = (x * torch.sin(x)**2) % np.pi  
        negative_output = (torch.rand_like(x) * x) % np.pi 
        return torch.where(x > 0, positive_output, negative_output)

class SchrodingersCatNN(nn.Module):
    """
    Neural network model representing Schrödinger's cat.

    Utilizes a custom probabilistic activation function to introduce quantum-like behavior.

    Attributes:
        fc1 (nn.Linear): First linear layer.
        bn1 (nn.BatchNorm1d): Batch normalization after the first layer.
        quantum_relu (ProbabilisticActivation): Custom activation function.
        fc2 (nn.Linear): Second linear layer.
        bn2 (nn.BatchNorm1d): Batch normalization after the second layer.
        fc3 (nn.Linear): Third linear layer.
        dropout (nn.Dropout): Dropout layer for regularization.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SchrodingersCatNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size) 
        self.quantum_relu = ProbabilisticActivation()
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)  
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)  
        self.fc3 = nn.Linear(hidden_size * 2, output_size)  

        self.dropout = nn.Dropout(0.3)  

    def forward(self, x):
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.quantum_relu(out)
        out = self.dropout(out) 

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.quantum_relu(out)
        out = self.dropout(out) 

        out = self.fc3(out)
        return out

class SchrodingersCatBox:
    """
    Represents a box containing two neural networks (alive and dead states).

    Implements quantum-inspired training and evaluation with wave function collapse.

    Attributes:
        alive_model (SchrodingersCatNN): Model representing the "alive" state.
        dead_model (SchrodingersCatNN): Model representing the "dead" state.
        alive_optimizer (optim.Optimizer): Optimizer for the alive model.
        dead_optimizer (optim.Optimizer): Optimizer for the dead model.
        superposition (bool): Indicates if the models are in superposition.
        collapsed_state (str): The final collapsed state of the system ("alive" or "dead").
    """
    def __init__(self, input_size, hidden_size, output_size):
        self.alive_model = SchrodingersCatNN(input_size, hidden_size, output_size)
        self.dead_model = SchrodingersCatNN(input_size, hidden_size, output_size)
        self.alive_model.apply(init_weights)
        self.dead_model.apply(init_weights)
        self.alive_optimizer = optim.Adam(self.alive_model.parameters(), lr=0.001)
        self.dead_optimizer = optim.Adam(self.dead_model.parameters(), lr=0.001)
        self.superposition = True

    def train_step(self, inputs, labels, criterion):
        """
        Performs a single training step on both models or the collapsed model.

        Args:
            inputs (torch.Tensor): Input tensor for the training step.
            labels (torch.Tensor): Ground truth labels.
            criterion (nn.Module): Loss function to compute the loss.

        Returns:
            float: The average loss if in superposition, or the loss of the collapsed model.
        """
        if self.superposition:
            # train both models
            self.alive_optimizer.zero_grad()
            alive_outputs = self.alive_model(inputs)
            alive_loss = criterion(alive_outputs, labels)
            alive_loss.backward()
            self.alive_optimizer.step()

            self.dead_optimizer.zero_grad()
            dead_outputs = self.dead_model(inputs)
            dead_loss = criterion(dead_outputs, labels)
            dead_loss.backward()
            self.dead_optimizer.step()

            return (alive_loss.item() + dead_loss.item()) / 2
        else:
            # train only the collapsed model
            model = self.alive_model if self.collapsed_state == "alive" else self.dead_model
            optimizer = self.alive_optimizer if self.collapsed_state == "alive" else self.dead_optimizer
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            return loss.item()

    def evaluate(self, dataloader, criterion):
        """
        Evaluates both models on the given dataloader.

        Args:
            dataloader (DataLoader): Dataloader for evaluation.
            criterion (nn.Module): Loss function for evaluation.

        Returns:
            list: A list of tuples containing accuracy and loss for each model.
        """
        performances = []
        for model in [self.alive_model, self.dead_model]:
            model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in dataloader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total
            performances.append((accuracy, avg_loss))
        return performances

    def collapse_wave_function(self, performances):
        """
        Collapses the wave function based on model performances.

        Determines the final state (alive or dead) probabilistically.

        Args:
            performances (list): List containing accuracy and loss for both models.
        """
        alive_performance, dead_performance = performances
        total_accuracy = alive_performance[0] + dead_performance[0]
        p_alive = alive_performance[0] / total_accuracy

        if quantum_random() < p_alive:
            self.collapsed_state = "alive"
            print(f"Wave function collapsed: Cat is alive with probability {p_alive:.2f}")
        else:
            self.collapsed_state = "dead"
            print(f"Wave function collapsed: Cat is dead with probability {1-p_alive:.2f}")

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, 
                          generator=torch.Generator().manual_seed(int(quantum_random() * 1e9)))
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

input_size = X_train.shape[1] 
hidden_size = 64
output_size = 2  
schrodingers_cat_box = SchrodingersCatBox(input_size, hidden_size, output_size)

criterion = DOALoss(alive_weight=1.0, dead_weight=1.0)

alive_counts = []
learned_counts = []

num_runs = 10
epochs = 20

for run in range(num_runs):
    print(f"Run {run+1}/{num_runs}")

    schrodingers_cat_box = SchrodingersCatBox(input_size, hidden_size, output_size)
    criterion = DOALoss(alive_weight=1.0, dead_weight=1.0)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            loss = schrodingers_cat_box.train_step(inputs, labels, criterion)
            total_loss += loss

        avg_loss = total_loss / len(train_loader) 
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    performances = schrodingers_cat_box.evaluate(test_loader, criterion)
    schrodingers_cat_box.collapse_wave_function(performances)

    if schrodingers_cat_box.collapsed_state == "alive":
        alive_counts.append(1)
    else:
        alive_counts.append(0)

    chosen_model = schrodingers_cat_box.alive_model if schrodingers_cat_box.collapsed_state == "alive" else schrodingers_cat_box.dead_model
    chosen_model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = chosen_model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)

    print(f"Final Accuracy: {accuracy * 100:.2f}%, Final Loss: {avg_loss:.4f}")

    p_learned = accuracy * (1 - avg_loss)
    if quantum_random() < p_learned:
        learned_counts.append(1)
        print(f"Model has successfully learned (Collapse: Learned state) with probability {p_learned:.2f}")
    else:
        learned_counts.append(0)
        print(f"Model failed to learn (Collapse: Unlearned state) with probability {1 - p_learned:.2f}")

alive_percentage = np.mean(alive_counts) * 100
learned_percentage = np.mean(learned_counts) * 100

print(f"Cat is alive in {alive_percentage:.2f}% of runs")
print(f"Model learned in {learned_percentage:.2f}% of runs")