import pandas as pd
import numpy as np
import torch
import os
import sys
import pickle
import random
from transformers import DistilBertTokenizer
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy

# ======================================================================================
# PATH CONFIGURATION
# ======================================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, "backend", "data", "synthetic_val_1000.csv")
WEIGHTS_PATH = os.path.join(BASE_DIR, "backend", "output", "GroupGen_Encoder_Final_Safe.pt")
SCALER_PATH = os.path.join(BASE_DIR, "backend", "data", "standard_scaler.pkl")
OUTPUT_DOC = os.path.join(BASE_DIR, "backend", "output", "MultimodalGroup_output.txt")

# Path-aware imports
sys.path.append(os.path.join(BASE_DIR, "backend"))
sys.path.append(os.path.join(BASE_DIR, "backend", "src"))

from src.clustering import compute_feature_vector
from src.kmedoids import kmedoids_pam
from deep_learning.models.multimodal_autoencoder import GroupGenEncoder


def calculate_learning_style_entropy(group_df):
    """Calculate Shannon entropy of learning style distribution in a group."""
    style_counts = group_df['Learning_Style'].value_counts()
    total = len(group_df)
    probabilities = style_counts / total
    # Add small epsilon to avoid log(0)
    return float(entropy(probabilities))


def inspect_groups():
    audit_lines = []
    def log(line=""):
        print(line)
        audit_lines.append(line)

    log("="*80)
    log("CLASSROOM AUDIT: GroupGen-Encoder Semantic Manifold Verification")
    log("="*80)
    
    # Load data
    log(f"\nLoading validation dataset: {CSV_PATH}")
    full_df = pd.read_csv(CSV_PATH)
    
    # Random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Sample 30 students (simulating one classroom)
    sample_indices = np.random.choice(len(full_df), size=30, replace=False)
    classroom_df = full_df.iloc[sample_indices].reset_index(drop=True)
    
    log(f"Sampled {len(classroom_df)} students from pool of {len(full_df)}")
    
    # Load scaler and compute tabular features
    log(f"\nLoading scaler: {SCALER_PATH}")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    tabular_matrix = compute_feature_vector(classroom_df, load_scaler_path=SCALER_PATH)
    tabular_dim = tabular_matrix.shape[1]
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Loading model on {device}: {WEIGHTS_PATH}")
    model = GroupGenEncoder(tabular_input_dim=tabular_dim, freeze_text=True)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # Precompute text embeddings
    log("\nPrecomputing text embeddings...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokens = tokenizer(classroom_df['Text'].tolist(), padding='max_length', 
                       max_length=64, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        text_embeddings = model.text_branch(input_ids, attention_mask)
    
    # Get bottleneck embeddings
    log("Generating bottleneck embeddings...")
    tabular_tensor = torch.tensor(tabular_matrix, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        bottleneck = model.encode(tabular_x=tabular_tensor, precomputed_text=text_embeddings)
    
    bottleneck_np = bottleneck.cpu().numpy()
    
    # Cluster into 6 groups of 5
    log("\nClustering 30 students into 6 groups of 5...")
    dist_matrix = pairwise_distances(bottleneck_np, metric='manhattan')
    cluster_labels, _ = kmedoids_pam(dist_matrix, K=6, random_state=42)
    
    classroom_df['Cluster'] = cluster_labels
    
    # Calculate overall learning style entropy
    overall_entropy = calculate_learning_style_entropy(classroom_df)
    
    log("\n" + "="*80)
    log(f"CLASSROOM LEARNING STYLE ENTROPY: {overall_entropy:.4f}")
    log("(Higher entropy = more diverse learning styles)")
    log("="*80)
    
    # Print audit for each group
    for group_id in range(6):
        group_df = classroom_df[classroom_df['Cluster'] == group_id].reset_index(drop=True)
        group_entropy = calculate_learning_style_entropy(group_df)
        
        log(f"\n{'─'*80}")
        log(f"GROUP {group_id + 1} (n={len(group_df)} students) | Learning Style Entropy: {group_entropy:.4f}")
        log(f"{'─'*80}")
        
        # Create audit table
        audit_data = []
        for idx, row in group_df.iterrows():
            bio_snippet = row['Text'][:50] + "..." if len(row['Text']) > 50 else row['Text']
            audit_data.append({
                'ID': idx + 1,
                'Learning_Style': row['Learning_Style'],
                'Motivation': row['Motivation'],
                'Self_Esteem': row['Self_Esteem'],
                'Work_Ethic': row['Work_Ethic'],
                'Bio_Snippet': bio_snippet
            })
        
        audit_table = pd.DataFrame(audit_data)
        table_str = audit_table.to_string(index=False)
        log(table_str)
        log()

    # Write audit output to file
    os.makedirs(os.path.dirname(OUTPUT_DOC), exist_ok=True)
    with open(OUTPUT_DOC, 'w', encoding='utf-8') as f:
        f.write("\n".join(audit_lines))

    log(f"Audit output written to: {OUTPUT_DOC}")


if __name__ == "__main__":
    inspect_groups()
