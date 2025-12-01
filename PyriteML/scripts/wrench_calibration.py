import torch
import torch.nn as nn
import torch.optim as optim
import zarr
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
import os
from typing import Dict, Callable, Tuple, List

SCRIPT_PATH = "/home/yifanhou/git/PyriteML/scripts"
sys.path.append(os.path.join(SCRIPT_PATH, '../'))

from PyriteUtility.spatial_math import spatial_utilities as su
from PyriteUtility.plotting.traj_and_rgb import draw_timed_traj_and_rgb
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs
register_codecs()

dataset_path = "/shared_local/data/processed/online_stow_real_test/processed"

print("Loading dataset from: ", dataset_path)
# load the zarr dataset from the path
# ‘r’ means read only (must exist); ‘r+’ means read/write (must exist); ‘a’ means read/write (create if doesn’t exist); ‘w’ means create (overwrite if exists); ‘w-’ means create (fail if exists).
buffer = zarr.open(dataset_path, mode="r+")
ep_name = "episode_1744067569"

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=6):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

# Generate synthetic data for demonstration
# Replace this with your actual data loading
def generate_sample_data(n_samples=1000):
    # Create some random 6D input vectors
    X = np.random.randn(n_samples, 6)
    
    # Create a non-linear mapping for demonstration
    Y = np.zeros((n_samples, 6))
    Y[:, 0] = X[:, 0]**2 - 2*X[:, 1] + X[:, 2]*X[:, 3]
    Y[:, 1] = np.sin(X[:, 0]) + X[:, 4]**2
    Y[:, 2] = X[:, 1] + X[:, 3] - X[:, 5]
    Y[:, 3] = np.exp(X[:, 2]/5) + X[:, 4] - X[:, 5]
    Y[:, 4] = X[:, 0] * X[:, 5] + X[:, 2]
    Y[:, 5] = np.cos(X[:, 3]) + X[:, 1] * X[:, 4]
    
    return X, Y

def load_data_from_zarr():
    wrench_0 = np.array(buffer["data"][ep_name]["wrench_0"])
    robot_wrench_0 = np.array(buffer["data"][ep_name]["robot_wrench_0"])
    ati_0 = robot_wrench_0 - wrench_0

    return robot_wrench_0, ati_0

# Training function
def train_model(model, X_train, Y_train, X_val, Y_val, criterion, optimizer, 
                num_epochs=100, batch_size=648, early_stopping_patience=10):
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, Y_val).item()
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

# Function to evaluate model
def evaluate_model(model, X_test, Y_test, criterion):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, Y_test).item()
        
        # Calculate per-dimension MSE
        dim_mse = torch.mean((test_outputs - Y_test)**2, dim=0)
        
        # Calculate R^2 score per dimension
        y_mean = torch.mean(Y_test, dim=0)
        ss_tot = torch.sum((Y_test - y_mean.unsqueeze(0))**2, dim=0)
        ss_res = torch.sum((Y_test - test_outputs)**2, dim=0)
        r2_scores = 1 - (ss_res / ss_tot)
        
    return test_loss, dim_mse, r2_scores, test_outputs

# Plot training curves
def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.show()

# Plot predictions vs actual
def plot_predictions(Y_test, predictions):
    # Reshape to visualize dimension by dimension
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(6):
        axes[i].scatter(Y_test[:, i].cpu().numpy(), predictions[:, i].cpu().numpy(), alpha=0.5)
        axes[i].plot([Y_test[:, i].min().item(), Y_test[:, i].max().item()], 
                    [Y_test[:, i].min().item(), Y_test[:, i].max().item()], 
                    'r--')
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png')
    plt.show()

# Main function
def main():
    # Generate or load your data
    # X, Y = generate_sample_data(n_samples=5000)
    X, Y = load_data_from_zarr()
    
    # Split data into train, validation, and test sets
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)
    
    Y_train = scaler_Y.fit_transform(Y_train)
    Y_val = scaler_Y.transform(Y_val)
    Y_test = scaler_Y.transform(Y_test)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.FloatTensor(Y_train)
    X_val = torch.FloatTensor(X_val)
    Y_val = torch.FloatTensor(Y_val)
    X_test = torch.FloatTensor(X_test)
    Y_test = torch.FloatTensor(Y_test)
    
    # Hyperparameters
    input_size = 6
    hidden_size = 64
    output_size = 6
    learning_rate = 0.001
    num_epochs = 300
    batch_size = 64
    
    # Initialize model
    model = MLP(input_size, hidden_size, output_size)
    
    # Define loss function (MSE for least-squares error) and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    model, train_losses, val_losses = train_model(
        model, X_train, Y_train, X_val, Y_val, 
        criterion, optimizer, num_epochs, batch_size
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    # Evaluate the model
    test_loss, dim_mse, r2_scores, predictions = evaluate_model(model, X_test, Y_test, criterion)
    
    print(f"Test MSE: {test_loss:.6f}")
    print("Per-dimension MSE:")
    for i, mse in enumerate(dim_mse):
        print(f"  Dimension {i+1}: {mse:.6f}")
    
    print("Per-dimension R² scores:")
    for i, r2 in enumerate(r2_scores):
        print(f"  Dimension {i+1}: {r2:.6f}")
    
    # Plot predictions vs actual values
    plot_predictions(Y_test, predictions)
    
    # Save the model
    model_info = {
        "scaler_X": scaler_X,
        "scaler_Y": scaler_Y,
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
    }
    torch.save(model.state_dict(), "wrench_model.pth")
    filename = "wrench_model_info.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model_info, file)
    
    # Example of using the model for inference
    # Remember to apply the same preprocessing to new data
    def predict(model, X_new):
        X_new_scaled = torch.FloatTensor(scaler_X.transform(X_new))
        model.eval()
        with torch.no_grad():
            Y_pred_scaled = model(X_new_scaled)
            Y_pred = scaler_Y.inverse_transform(Y_pred_scaled.numpy())
        return Y_pred
    
    # Example usage
    X_new = np.random.randn(5, 6)  # 5 new samples to predict
    Y_pred = predict(model, X_new)
    print("\nPredictions for new samples:")
    print(Y_pred)

if __name__ == "__main__":
    main()