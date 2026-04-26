import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import pickle
from transformers import DistilBertTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as sklearn_shuffle

# ============================================================
# ENVIRONMENT DETECTION: Colab vs Local
# ============================================================
def detect_environment():
    """Detect if running in Google Colab or locally."""
    try:
        from google.colab import drive
        return 'colab'
    except ImportError:
        return 'local'

ENVIRONMENT = detect_environment()
print(f"Environment Detected: {ENVIRONMENT.upper()}")

# Auto-select device based on environment
if ENVIRONMENT == 'colab':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Colab Mode: Using device = {DEVICE}")
else:
    DEVICE = 'cpu'
    print(f"Local Mode: Using device = {DEVICE}")

# ============================================================
# HELPER FUNCTION: Compute Feature Vector (Standalone)
# ============================================================
def compute_feature_vector_standalone(df, save_scaler_path=None, load_scaler_path=None):
    """
    Standalone feature vector computation (no external imports from clustering.py).
    Converts student features to numerical vectors.
    """
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    
    # Extract numeric features
    numeric_features = df[['Motivation', 'Self_Esteem', 'Work_Ethic']].values
    
    # Encode Learning_Style
    one_hot_encoder = OneHotEncoder(categories=[['Visual', 'Auditory', 'Kinesthetic']], sparse_output=False, handle_unknown='ignore')
    learning_style_encoded = one_hot_encoder.fit_transform(df[['Learning_Style']])
    
    # Combine
    features = np.hstack([numeric_features, learning_style_encoded])
    
    # Normalize
    if load_scaler_path and os.path.exists(load_scaler_path):
        with open(load_scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        features = scaler.transform(features)
    else:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        if save_scaler_path:
            os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
            with open(save_scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
    
    return features

# ============================================================
# IMPORT MODEL (from local module)
# ============================================================
from multimodal_autoencoder import MultimodalAutoencoder

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def main():
    print("="*60)
    print("PHASE 1: Loading Golden Dataset...")
    
    # Path-agnostic: Look in current working directory
    csv_path = 'synthetic_multimodal_1000.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please ensure the CSV is in the current directory.")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} students from {csv_path}")
    
    print("Computing Tabular Features...")
    scaler_path = 'standard_scaler.pkl'
    tabular_matrix = compute_feature_vector_standalone(df, save_scaler_path=scaler_path)
    tabular_dim = tabular_matrix.shape[1]
    print(f"Tabular dimension: {tabular_dim}")
    
    print("Tokenizing Text Features (DistilBERT)...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokens = tokenizer(df['Text'].tolist(), padding='max_length', max_length=64, truncation=True, return_tensors='pt')
    
    print("\nPHASE 2: Initializing Deep Learning Autoencoder on device: {}".format(DEVICE))
    model = MultimodalAutoencoder(tabular_input_dim=tabular_dim, freeze_text=True)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion_tab = nn.MSELoss()
    criterion_text = nn.MSELoss()
    
    tabular_tensor = torch.tensor(tabular_matrix, dtype=torch.float32).to(DEVICE)
    
    epochs = 20
    batch_size = 100
    
    print("\nPHASE 3: Pre-computing Text Features (DistilBERT) to Accelerate Training...")
    print("="*60)
    model.eval()
    all_text_features = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            b_ids = tokens['input_ids'][i:i+batch_size].to(DEVICE)
            b_mask = tokens['attention_mask'][i:i+batch_size].to(DEVICE)
            features = model.text_branch(b_ids, b_mask)
            all_text_features.append(features)
    precomputed_text_tensor = torch.cat(all_text_features, dim=0)
    
    print("\nPHASE 4: Training Commencing. Watch the Loss Drop!")
    print("="*60)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_loss_tab = 0
        total_loss_text = 0
        for i in range(0, len(df), batch_size):
            b_tab = tabular_tensor[i:i+batch_size]
            b_text = precomputed_text_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            reconstructed_tab, reconstructed_text, bottleneck = model(tabular_x=b_tab, precomputed_text=b_text)
            loss_tab = criterion_tab(reconstructed_tab, b_tab)
            loss_text = criterion_text(reconstructed_text, b_text)
            loss = (10.0 * loss_tab) + (1.0 * loss_text)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_tab += loss_tab.item()
            total_loss_text += loss_text.item()
            
        avg_loss_tab = total_loss_tab / (len(df) // batch_size)
        avg_loss_text = total_loss_text / (len(df) // batch_size)
        print(f"Epoch [{epoch+1}/{epochs}] - Tabular MSE: {avg_loss_tab:.4f}, Text MSE: {avg_loss_text:.4f}, Weighted Loss: {total_loss/len(df):.4f}")
        
    print("\nTraining complete! Autoencoder successfully learned to compress features.")
    
    print("\nEvaluating Text Recovery (Cosine Similarity) on Random Batch...")
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        # Select a random batch
        random_start = np.random.randint(0, len(df) - batch_size)
        b_tab_rand = tabular_tensor[random_start:random_start+batch_size]
        b_text_rand = precomputed_text_tensor[random_start:random_start+batch_size]
        _, reconstructed_text_rand, _ = model(tabular_x=b_tab_rand, precomputed_text=b_text_rand)
        cos_sim = F.cosine_similarity(b_text_rand, reconstructed_text_rand, dim=1).mean().item()
    print(f"Average Text Recovery Cosine Similarity (Random Batch): {cos_sim:.4f}")
    
    # Save model weights to current directory
    output_path = 'multimodal_autoencoder_final.pt'
    torch.save(model.state_dict(), output_path)
    print(f"\n✅ SUCCESS! Model weights saved to: {output_path}")

if __name__ == "__main__":
    main()
