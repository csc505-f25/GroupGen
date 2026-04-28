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

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to GroupGen root

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
            # THE PERMANENT FIX: Only attempt to create a directory if a path is provided
            dir_name = os.path.dirname(save_scaler_path)
            if dir_name: 
                os.makedirs(dir_name, exist_ok=True)
            
            with open(save_scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
    
    return features

# ============================================================
# IMPORT MODEL (from local module)
# ============================================================
sys.path.insert(0, os.path.join(BASE_DIR, 'backend', 'deep_learning', 'models'))
from multimodal_autoencoder import GroupGenEncoder

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def main():
    print("="*60)
    print("PHASE 1: Loading Leakage-Safe Datasets...")
    
    # 1. Load the separate files (Ensure these are uploaded to Colab)
    train_path = os.path.join(BASE_DIR, 'backend', 'data', 'synthetic_train_4000.csv')
    val_path = os.path.join(BASE_DIR, 'backend', 'data', 'synthetic_val_1000.csv')
    
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise FileNotFoundError("Safe datasets not found. Ensure train_4000 and val_1000 are uploaded.")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    print(f"Loaded {len(train_df)} Training students and {len(val_df)} Validation students.")

    # 2. Compute features using a SHARED scaler
    # We fit only on Train to prevent 'Data Snooping'
    scaler_path = os.path.join(BASE_DIR, "backend", "data", "standard_scaler.pkl")
    train_tab_matrix = compute_feature_vector_standalone(train_df, save_scaler_path=scaler_path)
    val_tab_matrix = compute_feature_vector_standalone(val_df, load_scaler_path=scaler_path)
    
    tabular_dim = train_tab_matrix.shape[1]
    
    # 3. Tokenize both sets
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_tokens = tokenizer(train_df['Text'].tolist(), padding='max_length', max_length=64, truncation=True, return_tensors='pt')
    val_tokens = tokenizer(val_df['Text'].tolist(), padding='max_length', max_length=64, truncation=True, return_tensors='pt')

    print("\nPHASE 2: Initializing GroupGen-Encoder on device: {}".format(DEVICE))
    model = GroupGenEncoder(tabular_input_dim=tabular_dim, freeze_text=True)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion_tab = nn.MSELoss()
    criterion_text = nn.MSELoss()

    # 4. Pre-compute Text for BOTH sets (Efficiency)
    print("\nPHASE 3: Pre-computing Text Features...")
    def precompute(tokens_obj, dataframe):
        model.eval()
        features_list = []
        with torch.no_grad():
            for i in range(0, len(dataframe), 100):
                ids = tokens_obj['input_ids'][i:i+100].to(DEVICE)
                mask = tokens_obj['attention_mask'][i:i+100].to(DEVICE)
                f = model.text_branch(ids, mask)
                features_list.append(f)
        return torch.cat(features_list, dim=0)

    precomputed_train_text = precompute(train_tokens, train_df)
    precomputed_val_text = precompute(val_tokens, val_df)

    # 5. Conversion to Tensors
    train_tab_tensor = torch.tensor(train_tab_matrix, dtype=torch.float32).to(DEVICE)
    val_tab_tensor = torch.tensor(val_tab_matrix, dtype=torch.float32).to(DEVICE)

    # 6. Training Loop with Validation Check
    print("\nPHASE 4: Training Commencing (Leakage-Safe Mode)")
    print("="*60)
    
    best_val_loss = float('inf')
    epochs = 30 # Increased slightly for the new split
    batch_size = 100

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        # Batch Training
        for i in range(0, len(train_df), batch_size):
            b_tab = train_tab_tensor[i:i+batch_size]
            b_text = precomputed_train_text[i:i+batch_size]
            
            optimizer.zero_grad()
            re_tab, re_text, _ = model(tabular_x=b_tab, precomputed_text=b_text)
            
            l_tab = criterion_tab(re_tab, b_tab)
            l_text = criterion_text(re_text, b_text)
            loss = (10.0 * l_tab) + (1.0 * l_text)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # 7. Validation Phase (The "Truth" check)
        model.eval()
        with torch.no_grad():
            v_re_tab, v_re_text, _ = model(tabular_x=val_tab_tensor, precomputed_text=precomputed_val_text)
            v_l_tab = criterion_tab(v_re_tab, val_tab_tensor)
            v_l_text = criterion_text(v_re_text, precomputed_val_text)
            val_loss = (10.0 * v_l_tab) + (1.0 * v_l_text)
            
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {total_train_loss/len(train_df):.4f} | Val Loss: {val_loss/len(val_df):.4f}")
        
        # Save the BEST model based on Validation performance
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "backend", "output", "GroupGen_Encoder_Final_Safe.pt"))
            print(" -> New Best Model Saved!")

if __name__ == "__main__":
    main()
