import pandas as pd
import numpy as np
import torch
import os
import sys
import pickle
import random
from transformers import DistilBertTokenizer
from sklearn.metrics import silhouette_score, pairwise_distances, calinski_harabasz_score, davies_bouldin_score
from torch.utils.data import DataLoader, TensorDataset

# ======================================================================================
# 1. PATH CONFIGURATION (Hard-coded for perfection)
# ======================================================================================
BASE_DIR = r"C:\Users\labuser.DESKTOP-3O6S6S6\Documents\GroupGen"
CSV_PATH = os.path.join(BASE_DIR, "backend", "data", "synthetic_val_1000.csv")
WEIGHTS_PATH = os.path.join(BASE_DIR, "backend", "output", "GroupGen_Encoder_Final_Safe.pt")
SCALER_PATH = os.path.join(BASE_DIR, "backend", "data", "standard_scaler.pkl")

# Path-aware imports
sys.path.append(os.path.join(BASE_DIR, "backend"))
sys.path.append(os.path.join(BASE_DIR, "backend", "src"))

from src.clustering import compute_feature_vector
from src.kmedoids import kmedoids_pam
from deep_learning.models.multimodal_autoencoder import GroupGenEncoder
import gower

def main():
    print(f"Loading UNSEEN Validation Pool: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing validation data! Ensure {CSV_PATH} exists.")
    
    full_df = pd.read_csv(CSV_PATH)
    
    print("Loading Global Scaler (Translation Key)...")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
        
    # Standard Scaler from Colab ensures the evaluation data 'looks' like training data
    global_tabular_matrix = compute_feature_vector(full_df, load_scaler_path=SCALER_PATH)
    tabular_dim = global_tabular_matrix.shape[1]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading 'Safe' Model on {device}...")
    model = GroupGenEncoder(tabular_input_dim=tabular_dim, freeze_text=True)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    
    # ==================================================================================
    # 2. PRE-COMPUTATION (Maximizing Speed)
    # ==================================================================================
    print("Pre-computing Global Bottlenecks for 1000 students...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokens = tokenizer(full_df['Text'].tolist(), padding='max_length', max_length=64, truncation=True, return_tensors='pt')
    
    dataset = TensorDataset(torch.tensor(global_tabular_matrix, dtype=torch.float32), tokens['input_ids'], tokens['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    global_bottlenecks = []
    with torch.no_grad():
        for b_tab, b_ids, b_mask in dataloader:
            b_neck = model.encode(b_tab.to(device), b_ids.to(device), b_mask.to(device))
            global_bottlenecks.append(b_neck.cpu())
            
    global_bottlenecks = torch.cat(global_bottlenecks).numpy()
    
    # ==================================================================================
    # 3. MONTE CARLO SIMULATION
    # ==================================================================================
    classroom_sizes = [30, 40, 50, 60]
    num_trials = 30
    models = ['Baseline', 'GroupGen-Encoder']
    metrics = ['Silhouette', 'CH Index', 'Davies-Bouldin']
    
    results = {size: {m: {metric: [] for metric in metrics} for m in models} for size in classroom_sizes}
    
    print("\nStarting Monte Carlo Simulation...")
    for size in classroom_sizes:
        print(f"EVALUATING N={size} (across {num_trials} trials)")
        for trial in range(num_trials):
            sample_indices = random.sample(range(len(full_df)), size)
            df_sample = full_df.iloc[sample_indices].reset_index(drop=True)
            
            desired_group_size = random.choice([3, 4, 5])
            n_groups = max(1, len(df_sample) // desired_group_size)
            
            tabular_matrix = global_tabular_matrix[sample_indices]
            bottleneck = global_bottlenecks[sample_indices]
            
            # Evaluate Baseline
            dist_baseline = gower.gower_matrix(df_sample.drop(columns=['Text']))
            labels_base, _ = kmedoids_pam(dist_baseline, K=n_groups)
            
            # Evaluate GroupGen-Encoder (Option A)
            dist_early = pairwise_distances(bottleneck, metric='manhattan')
            labels_early, _ = kmedoids_pam(dist_early, K=n_groups)
            
            # Metrics Calculation
            if len(set(labels_base)) > 1:
                results[size]['Baseline']['Silhouette'].append(silhouette_score(dist_baseline, labels_base, metric='precomputed'))
                results[size]['Baseline']['CH Index'].append(calinski_harabasz_score(tabular_matrix, labels_base))
                results[size]['Baseline']['Davies-Bouldin'].append(davies_bouldin_score(tabular_matrix, labels_base))

            if len(set(labels_early)) > 1:
                results[size]['GroupGen-Encoder']['Silhouette'].append(silhouette_score(dist_early, labels_early, metric='precomputed'))
                results[size]['GroupGen-Encoder']['CH Index'].append(calinski_harabasz_score(bottleneck, labels_early))
                results[size]['GroupGen-Encoder']['Davies-Bouldin'].append(davies_bouldin_score(bottleneck, labels_early))

    # ==================================================================================
    # 4. FINAL REPORTING
    # ==================================================================================
    report_lines = ["\n" + "="*95, "FINAL ROBUSTNESS REPORT: LEAKAGE-FREE VALIDATION (N=1000 UNSEEN STUDENTS)", "="*95]
    report_lines.append(f"{'Size':<6} | {'Metric':<16} | {'Baseline (Gower)':<30} | {'GroupGen-Encoder (Manhattan)':<30}")
    report_lines.append("-" * 95)
    
    for size in classroom_sizes:
        for i, metric in enumerate(metrics):
            avg_base, std_base = np.mean(results[size]['Baseline'][metric]), np.std(results[size]['Baseline'][metric])
            avg_early, std_early = np.mean(results[size]['GroupGen-Encoder'][metric]), np.std(results[size]['GroupGen-Encoder'][metric])
            
            row = f"N={size:<4} | {metric:<16} | {avg_base:.4f} ± {std_base:.4f}          | {avg_early:.4f} ± {std_early:.4f}"
            report_lines.append(row if i == 0 else f"{'':<6} | {metric:<16} | {avg_base:.4f} ± {std_base:.4f}          | {avg_early:.4f} ± {std_early:.4f}")
        report_lines.append("-" * 95)
        
    final_text = "\n".join(report_lines)
    print(final_text)
    
    summary_path = os.path.join(BASE_DIR, "backend", "output", "final_report", "robustness_metrics_summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"\n✅ Perfect! Final results saved to {summary_path}")

if __name__ == "__main__":
    main()