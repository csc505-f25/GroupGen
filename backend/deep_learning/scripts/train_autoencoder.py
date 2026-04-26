import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import the model natively (expected in same directory in Colab)
from multimodal_autoencoder import MultimodalAutoencoder

# ============================================================
# ENVIRONMENT DETECTION: Colab vs Local
# ============================================================
def detect_environment():
    try:
        from google.colab import drive
        return 'colab'
    except ImportError:
        return 'local'

ENVIRONMENT = detect_environment()
print(f"Environment Detected: {ENVIRONMENT.upper()}")

if ENVIRONMENT == 'colab':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Colab Mode: Using device = {device}")
else:
    device = torch.device('cpu')
    print(f"Local Mode: Using device = {device}")

# ============================================================
# HELPER FUNCTION: Compute Feature Vector (Standalone)
# ============================================================
def compute_feature_vector_standalone(df, save_scaler_path='standard_scaler.pkl'):
    numeric_features = df[['Motivation', 'Self_Esteem', 'Work_Ethic']].values
    one_hot_encoder = OneHotEncoder(categories=[['Visual', 'Auditory', 'Kinesthetic']], sparse_output=False, handle_unknown='ignore')
    learning_style_encoded = one_hot_encoder.fit_transform(df[['Learning_Style']])
    features = np.hstack([numeric_features, learning_style_encoded])
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    with open(save_scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
            
    return features

# -------------------------------------------------------------------
# 1. Dataset Loader
# -------------------------------------------------------------------
class MultimodalStudentDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name="distilbert-base-uncased", max_length=128):
        self.df = pd.read_csv(csv_file)
        
        print("Vectorizing tabular data and saving scaler...")
        # Get purely numerical array using standalone function
        self.tabular_features = compute_feature_vector_standalone(self.df).astype(np.float32)
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.texts = self.df['Text'].fillna("").tolist()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 1. Grab Tabular Vector
        tab_x = torch.tensor(self.tabular_features[idx])
        
        # 2. Tokenize Text String
        text_encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "tabular": tab_x,
            "input_ids": text_encoded["input_ids"].squeeze(0), # remove extra batch dim
            "attention_mask": text_encoded["attention_mask"].squeeze(0)
        }

# -------------------------------------------------------------------
# 2. Training Loop
# -------------------------------------------------------------------
def train_autoencoder(epochs=50, batch_size=32, lr=1e-3, bottleneck_dim=16):
    # Path-agnostic: Look in current working directory
    data_path = 'synthetic_multimodal_5000.csv'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please ensure it is in the current directory.")
        return

    # Load dataset
    print("Initializing tokenizers and datasets...")
    dataset = MultimodalStudentDataset(data_path)
    
    # 80/20 train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize autoencoder (freeze_text=True to speed up local training)
    tabular_dim = dataset.tabular_features.shape[1] # usually ~6
    model = MultimodalAutoencoder(tabular_input_dim=tabular_dim, bottleneck_dim=bottleneck_dim, freeze_text=True).to(device)
    
    # Loss & Optimizer
    criterion = nn.MSELoss()
    # We only optimize the unfrozen parameters (tabular branch, fusion layer, decoder)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Scheduler and Early Stopping setup
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # History tracking for plotting
    history_train_loss = []
    history_val_loss = []
    
    # Save path setup (Current Directory)
    save_path = 'multimodal_autoencoder_final.pt'

    # Training logic
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            tabular = batch["tabular"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            optimizer.zero_grad()
            
            reconstructed_tabular, _, bottleneck = model(tabular, input_ids, attention_mask)
            
            # We want the decoder to correctly rebuild the original tabular input 
            # based purely on the compressed mixed features!
            loss = criterion(reconstructed_tabular, tabular)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * tabular.size(0)
            progress_bar.set_postfix({'MSE': loss.item()})
            
        train_loss /= train_size
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tabular = batch["tabular"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                reconstructed_tabular, _, _ = model(tabular, input_ids, attention_mask)
                loss = criterion(reconstructed_tabular, tabular)
                val_loss += loss.item() * tabular.size(0)
                
        val_loss /= val_size
        print(f"\n--- Epoch {epoch+1} Results ---")
        print(f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
        
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        
        # Scheduler step and Early Stopping Check
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f" -> Best model saved with Val MSE: {val_loss:.4f}\n")
        else:
            patience_counter += 1
            print(f" -> No improvement. Patience: {patience_counter}/{patience}\n")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"SUCCESS: Finished training. Best weights saved to {save_path}")

    # Plotting MSE Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history_train_loss) + 1), history_train_loss, label='Train MSE', marker='o')
    plt.plot(range(1, len(history_val_loss) + 1), history_val_loss, label='Validation MSE', marker='s')
    plt.title('Autoencoder Training & Validation MSE Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = 'training_mse_loss.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved MSE loss graph to {plot_path}")

if __name__ == "__main__":
    train_autoencoder()
