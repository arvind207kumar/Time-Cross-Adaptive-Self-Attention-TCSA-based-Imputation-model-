import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import dataloader , dataset

import h5py

def load_from_h5(file_path):
    """
    Loads data from an HDF5 file.

    Args:
        file_path (str): The path to the HDF5 file.

    Returns:
        dict: The loaded data.
    """
    data = {}
    with h5py.File(file_path, "r") as hf:
        # Iterate through the groups (train, val, test)
        for group_name in hf.keys(): 
            data[group_name] = {}
            # Iterate through datasets within each group
            for dataset_name in hf[group_name].keys():  
                data[group_name][dataset_name] = hf[group_name][dataset_name][()] 
    return data

z = load_from_h5('D:\proj\Time series imputation\Airquality_DataSets\Air_time_series_dataset\Air_time_series_dataset.h5')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
train_dataloader = [
            {
                'X': torch.tensor(z['train']['X'][i], dtype=torch.float32),
              #  'missing_mask': torch.tensor(z['train']['missing_mask'][i], dtype=torch.float32),
              #  'delta_mask': torch.tensor(train_data['delta_mask'][i], dtype=torch.float32),
              #  'indicating_mask': torch.tensor(train_data['indicating_mask'][i], dtype=torch.float32),
               # 'X_hat': torch.tensor(train_data['X_hat'][i], dtype=torch.float32)
            }
            for i in range(len(z['train']['X']))
        ]

val_dataloader = [
            {
                'X': torch.tensor(z['val']['X'][i], dtype=torch.float32),
                'missing_mask': torch.tensor(z['val']['missing_mask'][i], dtype=torch.float32),
                'delta_mask': torch.tensor(z['val']['delta_mask'][i], dtype=torch.float32),
                'indicating_mask': torch.tensor(z['val']['indicating_mask'][i], dtype=torch.float32),
                'X_hat': torch.tensor(z['val']['X_hat'][i], dtype=torch.float32)
            }
            for i in range(len(z['val']['X']))
        ]
test_dataloader = [
            {
                'X': torch.tensor(z['test']['X'][i], dtype=torch.float32),
                'missing_mask': torch.tensor(z['test']['missing_mask'][i], dtype=torch.float32),
                'delta_mask': torch.tensor(z['test']['delta_mask'][i], dtype=torch.float32),
                'indicating_mask': torch.tensor(z['test']['indicating_mask'][i], dtype=torch.float32),
                'X_hat': torch.tensor(z['test']['X_hat'][i], dtype=torch.float32)
            }
            for i in range(len(z['test']['X']))
        ]

def train_model(model, dataloader, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            # Move data to the appropriate device
            for key in data:
                data[key] = data[key].to(device)

            optimizer.zero_grad()
            outputs = model.calc_loss(data)  # Forward pass and loss calculation
            loss = outputs["reconstruction_loss"]  # Get the reconstruction loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Optimize

            total_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")



def test_model(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for data in dataloader:
            for key in data:
                data[key] = data[key].to(device)

            outputs = model.calc_loss(data)  # Forward pass for testing
            loss = outputs["final_reconstruction_MAE"]  # Get the reconstruction loss
            total_loss += loss.item()
        
        print(f"Test Loss: {total_loss / len(dataloader):.4f}")




if __name__ == "__main__":
    # Load data from the .h5 file
    file_path = 'path/to/your/dataset.h5'  # Specify your dataset path
    

    # Set device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
     #  n_layers=2,d_time=100,d_feature=z['test']['X'] , d_model=64,d_inner=256,n_head=8 , d_k=64//8 , d_v=64,dropout=0.1
    # Hyperparameters
    n_layers = 2
    d_time =   z['test']['X'].shape[1]       #  time dimension
    d_feature = z['test']['X'].shape[-1]       # Example feature dimension
    d_model = 64        # Embedding dimension
    d_inner = 256        # Inner layer dimension
    n_head = 8           # Number of heads in multi-head attention
    d_k = 8            # Dimension of keys
    d_v = 8           # Dimension of values
    dropout = 0.1        # Dropout rate
    epochs = 20          # Number of training epochs
    learning_rate = 0.1 # Learning rate

    # Initialize the model, optimizer
    model = DualBranchModel(
        n_layers=n_layers,
        d_time=d_time,
        d_feature=d_feature,
        d_model=d_model,
        d_inner=d_inner,
        n_head=n_head,
        d_k=d_k,
        d_v=d_v,
        dropout=dropout,
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_dataloader, optimizer, device, epochs)


