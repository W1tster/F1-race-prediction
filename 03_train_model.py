import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Model definition (regression: predicts continuous position 1-20)
class F1DeepPredictor(nn.Module):
    
    def __init__(self, input_features=21):
        super(F1DeepPredictor, self).__init__()
        
        self.fc1 = nn.Linear(input_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, 1)  # Output: 1 neuron for regression
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

print("=" * 80)
print("F1 RACE PREDICTION - NEURAL NETWORK TRAINING (REGRESSION)")
print("=" * 80)

# Load training dataset
print("\n[1/8] Loading data...")
df = pd.read_csv('training_dataset.csv')
print(f"  Loaded {len(df)} examples")

# Prepare features (X) and target (y)
feature_cols = [col for col in df.columns if col not in 
                ['meeting_key', 'driver_name', 'finishing_position', 'points']]

X = df[feature_cols].values
y = df['finishing_position'].values.astype(float)  # Keep as continuous 1-20

# Handle missing values
print(f"  Checking for NaN values...")
nan_mask = np.isnan(X)
if nan_mask.any():
    print(f"  Found {nan_mask.sum()} NaN values - filling with column means")
    col_means = np.nanmean(X, axis=0)
    for i in range(X.shape[1]):
        X[nan_mask[:, i], i] = col_means[i]

print(f"  Input features: {len(feature_cols)}")
print(f"  Target: positions 1-20 (continuous)")

# Normalize features
print("\n[2/8] Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.clip(X_scaled, -5, 5)  # Clip outliers
print("  Done (mean=0, std=1, clipped to ±5)")

# Split data: 70% train, 10% validation, 20% test
print("\n[3/8] Splitting data...")
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)

print(f"  Train: {len(X_train)} examples")
print(f"  Val:   {len(X_val)} examples")
print(f"  Test:  {len(X_test)} examples")

# Create PyTorch data loaders
print("\n[4/8] Creating PyTorch datasets...")
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"  Batch size: 16")
print(f"  Training batches: {len(train_loader)}")

# Initialize model, optimizer, loss function
print("\n[5/8] Initializing neural network...")
model = F1DeepPredictor(input_features=len(feature_cols))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()  # Mean Squared Error for regression

total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params:,}")
print(f"  Learning rate: 0.001")
print(f"  Loss function: MSELoss (regression)")

# Training loop
print("\n[6/8] Training model...")
print("=" * 80)

num_epochs = 100
best_val_mae = float('inf')

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_mae = 0.0
    
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        train_loss += loss.item()
        train_mae += torch.abs(outputs - batch_y).sum().item()
    
    train_loss /= len(train_loader)
    train_mae /= len(train_dataset)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_within_1 = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            val_mae += torch.abs(outputs - batch_y).sum().item()
            
            # Track predictions within ±1 position
            predicted = torch.clamp(torch.round(outputs), 1, 20)
            val_within_1 += (torch.abs(predicted - batch_y) <= 1).sum().item()
    
    val_loss /= len(val_loader)
    val_mae /= len(val_dataset)
    val_within_1_pct = 100 * val_within_1 / len(val_dataset)
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
    
    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | Val ±1: {val_within_1_pct:.1f}%")

print("=" * 80)
print(f"Best validation MAE: {best_val_mae:.2f} positions")

# Test the model
print("\n[7/8] Testing model...")
model.eval()
test_correct = 0
test_total = 0
test_mae = 0.0
test_within_1 = 0
test_within_2 = 0
test_within_3 = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs).squeeze()
        predicted = torch.clamp(torch.round(outputs), 1, 20)
        
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        test_mae += torch.abs(outputs - labels).sum().item()
        
        errors = torch.abs(predicted - labels)
        test_within_1 += (errors <= 1).sum().item()
        test_within_2 += (errors <= 2).sum().item()
        test_within_3 += (errors <= 3).sum().item()

test_accuracy = 100 * test_correct / test_total
test_mae = test_mae / test_total
test_within_1_pct = 100 * test_within_1 / test_total
test_within_2_pct = 100 * test_within_2 / test_total
test_within_3_pct = 100 * test_within_3 / test_total

print(f"\n{'='*60}")
print(f"  TEST RESULTS:")
print(f"  Exact Accuracy: {test_accuracy:.2f}%")
print(f"  Mean Absolute Error (MAE): {test_mae:.2f} positions")
print(f"  Within ±1 position: {test_within_1_pct:.2f}%")
print(f"  Within ±2 positions: {test_within_2_pct:.2f}%")
print(f"  Within ±3 positions: {test_within_3_pct:.2f}%")
print(f"{'='*60}")

# Save trained model
print("\n[8/8] Saving model...")
torch.save({
    'model_state': model.state_dict(),
    'scaler': scaler,
    'features': feature_cols,
}, 'f1_model.pth')

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"MAE: {test_mae:.2f} positions | Within ±1: {test_within_1_pct:.1f}%")
print("Model saved to: f1_model.pth")
