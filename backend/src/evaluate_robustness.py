import pandas as pd
import numpy as np
import torch
import os
import sys
from transformers import DistilBertTokenizer
from sklearn.metrics import silhouette_score, pairwise_distances
import random

# Setup Paths to import adjacent modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.clustering import compute_feature_vector
from src.kmedoids import kmedoids_pam
from deep_learning.models.multimodal_autoencoder import MultimodalAutoencoder
import gower

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    csv_path = os.path.join(base_dir, 'backend', 'data', 'synthetic_multimodal_5000.csv')
    weights_path = os.path.join(base_dir, 'backend', 'output', 'multimodal_autoencoder_final.pt')
    
    print(f"Loading massive pool dataset from synthetic_multimodal_5000.csv...")
    full_df = pd.read_csv(csv_path)
    
    print("Initializing DistilBERT Tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    print("Pre-computing Global Tabular Matrix to prevent dimension mismatch...")
    scaler_path = os.path.join(base_dir, 'backend', 'output', 'standard_scaler.pkl')
    global_tabular_matrix = compute_feature_vector(full_df, load_scaler_path=scaler_path)
    tabular_dim = global_tabular_matrix.shape[1]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading trained Dual-Decoder Autoencoder on {device}...")
    model = MultimodalAutoencoder(tabular_input_dim=tabular_dim, freeze_text=True)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    
    classroom_sizes = [30, 40, 50, 60]
    num_trials = 5
    
    results = {size: {'Baseline': [], 'Early Fusion': []} for size in classroom_sizes}
    
    print("\nStarting Monte Carlo Robustness Simulation...")
    
    for size in classroom_sizes:
        print(f"\n{'='*60}")
        print(f"EVALUATING N={size} (across {num_trials} trials)")
        print(f"{'='*60}")
        
        for trial in range(num_trials):
            # 1. Randomly sample N student indices
            sample_indices = random.sample(range(len(full_df)), size)
            df_sample = full_df.iloc[sample_indices].reset_index(drop=True)
            
            # 2. Randomly pick group size between 3 and 5
            desired_group_size = random.choice([3, 4, 5])
            n_groups = max(1, len(df_sample) // desired_group_size)
            
            # 3. Compute Tabular and Text features for this sample
            tabular_matrix = global_tabular_matrix[sample_indices]
            tokens = tokenizer(df_sample['Text'].tolist(), padding='max_length', max_length=64, truncation=True, return_tensors='pt')
            
            with torch.no_grad():
                b_tab = torch.tensor(tabular_matrix, dtype=torch.float32).to(device)
                b_ids = tokens['input_ids'].to(device)
                b_mask = tokens['attention_mask'].to(device)
                
                bottleneck = model.encode(b_tab, b_ids, b_mask)
                
                bottleneck = bottleneck.cpu().numpy()
            
            # 4. Evaluate Baseline (Tabular Only)
            dist_baseline = gower.gower_matrix(df_sample.drop(columns=['Name', 'Text']))
            labels_base, _ = kmedoids_pam(dist_baseline, K=n_groups)
            sil_base = silhouette_score(dist_baseline, labels_base, metric='precomputed') if len(set(labels_base)) > 1 else 0
            
            # 5. Evaluate Early Fusion
            dist_early = pairwise_distances(bottleneck, metric='manhattan')
            labels_early, _ = kmedoids_pam(dist_early, K=n_groups)
            sil_early = silhouette_score(bottleneck, labels_early, metric='manhattan') if len(set(labels_early)) > 1 else 0
            
            results[size]['Baseline'].append(sil_base)
            results[size]['Early Fusion'].append(sil_early)
            
            print(f"  Trial {trial+1}/5 | Grp Size: {desired_group_size} | Base: {sil_base:.4f} | Early: {sil_early:.4f}")

    
    report_lines = []
    report_lines.append("\n\n" + "="*60)
    report_lines.append("FINAL ROBUSTNESS REPORT (Average Silhouette Score)")
    report_lines.append("="*60)
    report_lines.append(f"{'Size':<10} | {'Baseline':<15} | {'Early Fusion':<15}")
    report_lines.append("-" * 60)
    for size in classroom_sizes:
        avg_base = np.mean(results[size]['Baseline'])
        avg_early = np.mean(results[size]['Early Fusion'])
        report_lines.append(f"N={size:<8} | {avg_base:<15.4f} | {avg_early:<15.4f}")
        
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save the robustness summary to output folder
    summary_path = os.path.join(base_dir, 'backend', 'output', 'robustness_metrics_summary.txt')
    with open(summary_path, "w") as f:
        f.write(report_text + "\n")
    print(f"\nSimulation Complete! Robustness metrics successfully saved to {summary_path}")

if __name__ == "__main__":
    main()
