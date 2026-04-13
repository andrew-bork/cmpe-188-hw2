import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

import torchvision

VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LOG_INTERVAL=1
LEARNING_RATE = 0.0005
REGULARIZATION_RATE = 0.0001
TITLE = "Fashion MNIST (nn)"
OUTPUT_DIR = "output/nn_fashion_mnist"

TARGET_VALIDATION_ACC = 0.85
TARGET_TRAIN_ACC = 0.85
TARGET_F1 = 0.7

import sys

def eprint(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, file=sys.stderr, **kwargs)
    
def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "nn_fashion_mnist",
        "task_type": "neural",
        "input_type": "image",
        "output_type": "classification",
        "description": "Fit a neural network on Fashion MNIST using ADAM."
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
def get_device():
    """Get device (cuda/cpu)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(validation_split=0.2, batch_size=32, device=None):
    """Load data from the csv."""
    if device is None:
        device = get_device()
    
    
    transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5), (0.5))])
        
        
    train_dataset = torchvision.datasets.FashionMNIST(".data/fashion_mnist", True, download=True, transform=transform)
    val_dataset = torchvision.datasets.FashionMNIST(".data/fashion_mnist", False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class Model(nn.Module):
    def __init__(self, input_dim: int, output_dimension: int):
        super(Model, self).__init__()
        self.layers = [
            nn.Linear(input_dim, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, output_dimension),
        ]
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, 1)
        return self.model(x)

def train(model, train_loader, val_loader, device=None, epochs=100, lr=0.1, weight_decay=0.01):
    if device is None:
        device = get_device()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(epochs):
        total = 0
        epoch_loss = 0.0
        for (X_batch, y_batch) in tqdm(train_loader):
            # Move to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total += len(y_batch)
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total = 0
        total_correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
                predictions = torch.argmax(outputs, dim=1)
                total_correct += (predictions == y_batch).sum().item()
                total += len(y_batch)
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        model.train()
        
        # Print progress every 20 epochs
        if (epoch + 1) % LOG_INTERVAL == 0:
            eprint(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {100*total_correct/total:.2f}%")
    
    return train_losses, val_losses

def evaluate(model: nn.Module, data_loader: DataLoader, device=None):
    """Evaluate the model and return metrics."""
    if device is None:
        device = get_device()
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(data_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            all_preds.append(predictions.cpu())
            all_targets.append(y_batch.cpu())
    
    all_preds = torch.concat(all_preds)
    all_targets = torch.concat(all_targets)
    
    accuracy = (all_preds == all_targets).sum() / len(all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    
    
    metrics = {
        'loss': total_loss / len(all_preds),
        'accuracy': accuracy,
        'f1_macro': f1
    }
    
    return metrics

def predict(model, X, device=None):
    """Make predictions."""
    if device is None:
        device = get_device()
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor)
        return predictions.cpu().numpy()

def save_artifacts(model, train_losses, val_losses, output_dir="output", filename_prefix="nn_mnist"):
    """Save model artifacts and visualization."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model
    model_path = os.path.join(output_dir, f"{filename_prefix}_model.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save training history
    history_path = os.path.join(output_dir, f"{filename_prefix}_history.json")
    history = {
        'train_losses': [float(l) for l in train_losses],
        'val_losses': [float(l) for l in val_losses]
    }
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create figure
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 5))
    
    # Plot 2: Loss curves
    ax1.plot(train_losses, 'b-', linewidth=2, label='Train Loss')
    ax1.plot(val_losses, 'r--', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    eprint(f"Artifacts saved to {output_dir}")

def main():
    eprint("=" * 60)
    eprint(TITLE)
    eprint("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    eprint(f"Using device: {device}")
    
    # Create dataloaders
    eprint("\nCreating dataloaders...")
    train_loader, val_loader= make_dataloaders(
        validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, device=device
    )
    eprint(f"Training samples: {len(train_loader)}, Validation samples: {len(val_loader)}")
    
    # Build model
    eprint("\nBuilding model...")
    train_features, train_labels = next(iter(train_loader))
    input_dimension = torch.flatten(train_features, 1).shape[1]
    output_dimension = 10
    model = Model(input_dimension, output_dimension).to(device)
    eprint(f"Model architecture: {model}")

    # Train model
    eprint("\nTraining model...")
    train_losses, val_losses = train(
        model, train_loader, val_loader, 
        device=device,
        epochs=EPOCHS, 
        lr=LEARNING_RATE, 
        weight_decay=REGULARIZATION_RATE
    )
    
    # Evaluate on both splits
    eprint("\nEvaluating model...")
    train_metrics = evaluate(model, train_loader, device=device)
    val_metrics = evaluate(model, val_loader, device=device)
    
    eprint("\nMetrics:")
    eprint(f"  Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1_macro']:.4f}, Accuracy: {100*train_metrics['accuracy']:.2f}%")
    eprint(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1_macro']:.4f}, Accuracy: {100*val_metrics['accuracy']:.2f}%")
    
    # Save artifacts
    eprint("\nSaving artifacts...")
    save_artifacts(model, train_losses, val_losses, output_dir=OUTPUT_DIR, filename_prefix="")
    
    # Quality checks
    eprint("\n" + "=" * 60)
    eprint("Quality Checks:")
    eprint("=" * 60)
    
    checks_passed = True

    if train_metrics['accuracy'] > TARGET_TRAIN_ACC:
        eprint(f"✓ Train accuracy > {TARGET_TRAIN_ACC}: {train_metrics['accuracy']:.4f}")
    else:
        eprint(f"✗ Train accuracy > {TARGET_TRAIN_ACC}: {train_metrics['accuracy']:.4f}")
        checks_passed = False
    
    if val_metrics['accuracy'] > TARGET_VALIDATION_ACC:
        eprint(f"✓ Val accuracy > {TARGET_VALIDATION_ACC}: {val_metrics['accuracy']:.4f}")
    else:
        eprint(f"✗ Val accuracy > {TARGET_VALIDATION_ACC}: {val_metrics['accuracy']:.4f}")
        checks_passed = False
    
    if val_metrics['f1_macro'] > TARGET_F1:
        eprint(f"✓ Val f1_macro > {TARGET_F1}: {val_metrics['f1_macro']:.4f}")
    else:
        eprint(f"✗ Val f1_macro > {TARGET_F1}: {val_metrics['f1_macro']:.4f}")
        checks_passed = False
    
    # Final summary
    eprint("=" * 60)
    if checks_passed:
        eprint("PASS: All quality checks passed!")
    else:
        eprint("FAIL: Some quality checks failed!")
    eprint("=" * 60)
    
    # Exit with appropriate code
    return 0 if checks_passed else 1

if __name__ == '__main__':
    exit(main())