import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load datasets
train_data = np.load('data/METR-LA/train.npz')
test_data = np.load('data/METR-LA/test.npz')
val_data = np.load('data/METR-LA/val.npz')

# Extract datasets and reshape
def process_data(data):
    x, y = data['x'], data['y']
    x = x.reshape(x.shape[0], x.shape[1], -1)
    y = y.reshape(y.shape[0], y.shape[1], -1)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X_train_tensor, y_train_tensor = process_data(train_data)
X_test_tensor, y_test_tensor = process_data(test_data)
X_val_tensor, y_val_tensor = process_data(val_data)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64, shuffle=False)

# LSTM model
class CustomTrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CustomTrafficLSTM, self).__init__()
        self.num_layers = num_layers  # Store num_layers as an instance variable
        self.hidden_size = hidden_size  # Store hidden_size as an instance variable
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state for LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Passing the output of the last time step to the fully connected layer
        out = self.fc(out)

        return out

# Initialize the model
model = CustomTrafficLSTM(X_train_tensor.shape[-1], 50, 2, y_train_tensor.shape[-1])

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Assuming the following variables are defined: model, train_loader, val_loader, test_loader

criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()  # To compute Mean Absolute Error
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
best_val_loss = float('inf')
patience = 10
trigger_times = 0

train_mae = []
val_mae = []

for epoch in range(500):
    model.train()
    train_loss = 0
    train_mae_accum = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_mae_accum += mae_criterion(outputs, labels).item()

    train_mae.append(train_mae_accum / len(train_loader))

    val_loss = 0
    val_mae_accum = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            val_mae_accum += mae_criterion(outputs, labels).item()

    val_mae.append(val_mae_accum / len(val_loader))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break

    print(f'Epoch {epoch + 1}, Val Loss: {val_loss / len(val_loader):.4f}')

# Test evaluation
test_loss = 0
test_mae_accum = 0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        test_mae_accum += mae_criterion(outputs, labels).item()

test_mae = test_mae_accum / len(test_loader)
print(f'Test Loss: {test_loss / len(test_loader):.4f}')
print(f'Test MAE: {test_mae:.4f}')

# Plot MAE learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_mae, label='Train MAE', color='blue')
plt.plot(val_mae, label='Validation MAE', color='red')
plt.title('MAE Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)
plt.show()
