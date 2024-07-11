import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import os
import zipfile

# Load datasets
train_df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None)
test_df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None)
rul_df = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None)

# Set column names
column_names = ['id', 'cycle'] + ['setting1', 'setting2', 'setting3'] + ['s' + str(i) for i in range(1, 22)]
train_df.columns = column_names
test_df.columns = column_names
rul_df.columns = ['RUL']

# Calculate the RUL for the train set
train_df['RUL'] = train_df.groupby('id')['cycle'].transform(lambda x: x.max()) - train_df['cycle']

# Define normalization columns (without 'RUL' column)
cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL'])

# Data normalization
scaler = MinMaxScaler()
train_df[cols_normalize] = scaler.fit_transform(train_df[cols_normalize])
test_df[cols_normalize] = scaler.transform(test_df[cols_normalize])

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Dataset class
class TurbofanDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        unit_id = self.data.iloc[idx]['id']
        unit_data = self.data[self.data['id'] == unit_id]
        start = idx % (len(unit_data) - self.sequence_length + 1)
        sequence = unit_data.iloc[start:start + self.sequence_length].drop(columns=['id', 'cycle', 'RUL']).values
        target = unit_data.iloc[start + self.sequence_length - 1]['RUL']
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# Definition of LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Training settings
sequence_length = 50
input_dim = len(cols_normalize)  # Number of input features
hidden_dim = 128  # Number of hidden units
num_layers = 3  # Number of LSTM layers
output_dim = 1  # Output dimension
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model initialization and training settings
model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Training data
train_dataset = TurbofanDataset(train_df, sequence_length)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=4)

# Model training
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, target_r2_score=0.82):
    train_losses = []
    val_losses = []
    val_r2_scores = []
    early_stopping_patience = 10
    early_stopping_counter = 0
    best_val_loss = float('inf')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for sequences, targets in train_loader:
            optimizer.zero_grad()
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                val_loss += criterion(outputs.squeeze(), targets).item()
                val_predictions.extend(outputs.squeeze().cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        val_r2 = r2_score(val_targets, val_predictions)
        val_r2_scores.append(val_r2)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R2 Score: {val_r2:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'model.pth')  # Save the best model
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        if val_r2 >= target_r2_score:
            print(f"Stopping training early as R² score reached {val_r2:.4f} at epoch {epoch + 1}")
            break

    return train_losses, val_losses, val_r2_scores

num_epochs = 100
target_r2_score = 0.82
train_losses, val_losses, val_r2_scores = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, target_r2_score)

def plot_training_history(train_losses, val_losses, val_r2_scores):
    epochs = range(1, len(train_losses) + 1)

    # Plotting training and validation loss
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting validation R2 score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_r2_scores, 'g-', label='Validation R2 Score')
    plt.title('Validation R2 Score')
    plt.xlabel('Epochs')
    plt.ylabel('R2 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

plot_training_history(train_losses, val_losses, val_r2_scores)

# Create a zip file containing model.pth and training_history.png
with zipfile.ZipFile('result.zip', 'w') as zipf:
    zipf.write('model.pth')
    zipf.write('training_history.png')

# Cleanup
os.remove('model.pth')
os.remove('training_history.png')

print("Zipped the model and training history plot into result.zip")
