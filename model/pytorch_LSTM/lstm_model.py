import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load datasets
train_data = np.load('data/METR-LA/test.npz')
test_data = np.load('data/METR-LA/test.npz')
val_data = np.load('data/METR-LA/val.npz')

print(list(train_data.keys()))

# Extracting datasets
X_train, y_train = train_data['x'], train_data['y']
X_test, y_test = test_data['x'], test_data['y']
X_val, y_val = val_data['x'], val_data['y']

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Checking the data dimensions
print("Original shape:", X_train_tensor.shape)

# DataLoader setup
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)

import torch.optim as optim
import torch.nn as nn

import torch.nn as nn

import torch
import torch.nn as nn


class CustomTrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CustomTrafficLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state for LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Passing the output of the last time step to the fully connected layer
        out = self.fc(out[:, -1, :])
        return out


# Parameters for model initialization
input_size = X_train.shape[-1]  # Assuming 'X_train' is defined in the data loading step
hidden_size = 50
num_layers = 2
output_size = y_train.shape[-1]  # Assuming 'y_train' is defined in the data loading step

# Initialize the LSTM model
model = CustomTrafficLSTM(input_size, hidden_size, num_layers, output_size)

# Print model architecture
print(model)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
patience = 10
best_val_loss = float('inf')
trigger_times = 0

# Training loop
for epoch in range(500):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation loss
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Print training and validation loss
    print(
        f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break

# Evaluate the model on the test set
test_loss = 0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()

print(f'Test Loss: {test_loss / len(test_loader):.4f}')
