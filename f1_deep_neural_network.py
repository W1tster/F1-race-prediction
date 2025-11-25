import torch
import torch.nn as nn
import torch.nn.functional as F

# Neural network model for F1 race position prediction
class F1DeepPredictor(nn.Module):
    
    def __init__(self, input_features=45):
        super(F1DeepPredictor, self).__init__()
        
        # Layer 1: Input -> 128
        self.fc1 = nn.Linear(input_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        # Layer 2: 128 -> 64
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        # Layer 3: 64 -> 32
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        
        # Output: 32 -> 1 (predicted position)
        self.fc4 = nn.Linear(32, 1)

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


# Initialize model with optimizer and loss function
def create_model_and_optimizer(input_features=45, learning_rate=0.001):
    model = F1DeepPredictor(input_features=input_features)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, criterion


# Train the model on F1 data
def train_model(model, train_loader, val_loader, optimizer, criterion, 
                num_epochs=100, device='cpu'):
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += batch_targets.size(0)
                correct += (predicted == batch_targets).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val Accuracy: {val_accuracy:.2f}%')
    
    return history


# Predict race positions for drivers
def predict_race_positions(model, features, device='cpu'):
    model.eval()
    model = model.to(device)
    features = features.to(device)
    
    with torch.no_grad():
        outputs = model(features)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted_positions = torch.max(outputs, 1)
        predicted_positions = predicted_positions + 1
    
    return predicted_positions.cpu(), probabilities.cpu()


# Example usage with dummy data
if __name__ == "__main__":
    print("=" * 80)
    print("F1 DEEP NEURAL NETWORK - COMPLETE EXAMPLE")
    print("=" * 80)
    
    print("\nStep 1: Creating dummy training data...")
    
    num_train = 800
    num_val = 200
    input_features = 45
    
    X_train = torch.randn(num_train, input_features)
    y_train = torch.randint(0, 20, (num_train,))
    
    X_val = torch.randn(num_val, input_features)
    y_val = torch.randint(0, 20, (num_val,))
    
    print(f"  Training samples: {num_train} drivers")
    print(f"  Validation samples: {num_val} drivers")
    print(f"  Features per driver: {input_features}")
    
    print("\nStep 2: Creating data loaders...")
    
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    
    print(f"  Batch size: 10 drivers per batch")
    print(f"  Training batches per epoch: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    print("\nStep 3: Initializing neural network...")
    
    model, optimizer, criterion = create_model_and_optimizer(
        input_features=45,
        learning_rate=0.001
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Neural network created with {total_params:,} parameters")
    
    print("\nStep 4: Training would happen here...")
    print("  (Skipping actual training in this example)")
    
    print("\nStep 5: Making predictions for 20 drivers...")
    
    race_features = torch.randn(20, 45)
    predictions, probabilities = predict_race_positions(model, race_features)
    
    print("\n  Predicted finishing positions:")
    for i, pos in enumerate(predictions[:10]):
        confidence = probabilities[i, pos-1].item() * 100
        print(f"    Driver {i+1}: Predicted P{pos.item()} ({confidence:.1f}% confidence)")
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE!")
    print("=" * 80)
