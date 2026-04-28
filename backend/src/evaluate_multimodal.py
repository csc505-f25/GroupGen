import pandas as pd
import numpy as np
import torch
import os
import sys
from transformers import DistilBertTokenizer
from sklearn.metrics import pairwise_distances

# Setup Paths to import adjacent modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from .clustering import compute_feature_vector, visualize_clustering, enforce_group_size
from .kmedoids import kmedoids_pam
from deep_learning.models.multimodal_autoencoder import GroupGenEncoder

def compare_multimodal_architectures(csv_path, weights_path, desired_group_size=4):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    n_groups = max(1, len(df) // desired_group_size)

    # 1. BASELINE: Extract standard Tabular Features
    print("Computing Baseline Tabular Matrix...")
    scaler_path = os.path.join(os.path.dirname(os.path.dirname(weights_path)), 'data', 'standard_scaler.pkl')
    tabular_matrix = compute_feature_vector(df, load_scaler_path=scaler_path)
    tabular_dim = tabular_matrix.shape[1]

    # 2. Extract Deep Learning Features
    print("Loading DistilBERT Tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    texts = df['Text'].tolist() if 'Text' in df.columns else [""] * len(df)
    tokens = tokenizer(texts, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
    
    print(f"Loading Trained GroupGenEncoder weights from {weights_path}...")
    model = GroupGenEncoder(tabular_input_dim=tabular_dim, freeze_text=True)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    
    print("Forward Pass: Extracting architectural representations in batches...")
    bottleneck_list = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing deep learning inference on: {device.type.upper()}")
    model = model.to(device)
    
    with torch.no_grad():
        tabular_tensor = torch.tensor(tabular_matrix, dtype=torch.float32)
        batch_size = 100
        for i in range(0, len(df), batch_size):
            end_idx = min(i + batch_size, len(df))
            
            b_tab = tabular_tensor[i:end_idx].to(device)
            b_ids = tokens['input_ids'][i:end_idx].to(device)
            b_mask = tokens['attention_mask'][i:end_idx].to(device)
            
            b = model.encode(b_tab, b_ids, b_mask)
            bottleneck_list.append(b.cpu().numpy())
            
        bottleneck = np.vstack(bottleneck_list)

    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    results = {}

    def evaluate_model(name, feature_matrix, labels, metric='euclidean'):
        try:
            sil = silhouette_score(feature_matrix, labels, metric=metric)
            ch = calinski_harabasz_score(feature_matrix, labels)
            db = davies_bouldin_score(feature_matrix, labels)
            results[name] = {"Silhouette": sil, "CH Index": ch, "Davies-Bouldin": db}
        except Exception as e:
            results[name] = {"Silhouette": str(e), "CH Index": "Error", "Davies-Bouldin": "Error"}

    # =========================================================================
    # ARCHITECTURE 1: Baseline (Tabular Only)
    # =========================================================================
    print("\n[1/3] Running BASELINE (K-Medoids Manhattan)...")
    dist_baseline = pairwise_distances(tabular_matrix, metric='manhattan')
    labels_baseline_raw, _ = kmedoids_pam(dist_baseline, K=n_groups, random_state=42)
    labels_baseline = enforce_group_size(labels_baseline_raw, desired_group_size, feature_matrix=tabular_matrix, expected_n_clusters=n_groups)
    
    evaluate_model("Baseline (Tabular)", tabular_matrix, labels_baseline, metric='manhattan')
    visualize_clustering(tabular_matrix, labels_baseline, n_groups, df=None, metric=f"Baseline Tabular N={len(df)}")

    # =========================================================================
    # ARCHITECTURE A: Early Fusion (16-D Bottleneck)
    # =========================================================================
    print("\n[2/3] Running OPTION A (Early Fusion Bottleneck)...")
    dist_early = pairwise_distances(bottleneck, metric='manhattan')
    labels_early_raw, _ = kmedoids_pam(dist_early, K=n_groups, random_state=42)
    labels_early = enforce_group_size(labels_early_raw, desired_group_size, feature_matrix=bottleneck, expected_n_clusters=n_groups)
    
    evaluate_model("Early Fusion (16-D)", bottleneck, labels_early, metric='manhattan')
    visualize_clustering(bottleneck, labels_early, n_groups, df=None, metric=f"16-D Early Fusion N={len(df)}")

    
    
    report = [
        f"\n{'='*65}",
        f"EVALUATION METRICS SUMMARY (N = {len(df)})",
        f"{'='*65}",
        f"{'Architecture':<25} | {'Silhouette':<10} | {'CH Index':<10} | {'Davies-Bouldin'}",
        "-" * 65
    ]
    for name, metrics in results.items():
        sil = metrics['Silhouette']
        ch_idx = metrics['CH Index']
        db_idx = metrics['Davies-Bouldin']
        
        sil_str = f"{sil:.4f}" if isinstance(sil, float) else "N/A"
        ch_str = f"{ch_idx:.1f}" if isinstance(ch_idx, float) else "N/A"
        db_str = f"{db_idx:.4f}" if isinstance(db_idx, float) else "N/A"
        report.append(f"{name:<25} | {sil_str:<10} | {ch_str:<10} | {db_str}")
        
    report.append("="*65)
    report_text = "\n".join(report)
    print(report_text)
    
    # Save it to a file so it isn't lost
    summary_path = os.path.join(os.path.dirname(weights_path), "metrics_summary.txt")
    with open(summary_path, "a") as f:
        f.write(report_text + "\n")
        
    print("Higher Silhouette/CH is better. Lower Davies-Bouldin is better.\n")
    print("SUCCESS! Baseline and Early Fusion architectures evaluated and visualized.")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    weights_file = os.path.join(base_dir, 'backend', 'output', 'GroupGen_Encoder_Final_Safe.pt')
    
    # Load the master 4000-student pool
    pool_csv = os.path.join(base_dir, 'backend', 'data', 'synthetic_train_4000.csv')
    pool_df = pd.read_csv(pool_csv)
    
    for n_size in [30, 40, 50, 60]:
        print(f"\n\n{'='*70}")
        print(f"RUNNING EXPERIMENT FOR N = {n_size} STUDENTS")
        print(f"{'='*70}\n")
        
        # Save a temporary static subset so the function can load it
        subset_df = pool_df.sample(n=n_size, random_state=42).reset_index(drop=True)
        subset_csv = os.path.join(base_dir, 'backend', 'data', f'temp_n_{n_size}.csv')
        subset_df.to_csv(subset_csv, index=False)
        
        compare_multimodal_architectures(subset_csv, weights_file)
        
        # Clean up the temporary CSV
        if os.path.exists(subset_csv):
            os.remove(subset_csv)
