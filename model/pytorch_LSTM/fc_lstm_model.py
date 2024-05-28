import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load datasets
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


# Define the FC-LSTM model
class FCLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(FCLSTM, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Encoder
        _, (hidden, cell) = self.encoder(x)
        # Decoder
        decoder_outputs, _ = self.decoder(x, (hidden, cell))
        # Fully connected layer
        output = self.fc(decoder_outputs)
        return output


# Hyperparameters
input_size = X_train_tensor.shape[2]  # Number of features
hidden_size = 256  # Number of LSTM units
output_size = y_train_tensor.shape[2]  # Output size
num_layers = 2  # Number of recurrent layers
batch_size = 64
learning_rate = 1e-4
l1_weight_decay = 2e-5
l2_weight_decay = 5e-4
num_epochs = 50  # Total number of epochs
early_stop_patience = 10  # Early stopping patience

# Initialize the model, loss function, and optimizer
model = FCLSTM(input_size, hidden_size, output_size, num_layers)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_weight_decay)

# Training loop with early stopping
best_val_loss = float('inf')
patience = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # L1 regularization
        l1_regularization = 0
        for param in model.parameters():
            l1_regularization += torch.sum(torch.abs(param))
        loss += l1_weight_decay * l1_regularization

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 0
    else:
        patience += 1
        if patience >= early_stop_patience:
            print("Early stopping")
            break

    # Learning rate decay
    if (epoch + 1) % 10 == 0 and (epoch + 1) >= 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

print("Training complete")
# Testing phase
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)

test_loss /= len(test_loader.dataset)
print(f'Test MAE: {test_loss:.4f}')