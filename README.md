# GroupGen-Encoder: Multimodal Joint Latent Manifold for Student Grouping

GroupGen-Encoder moves beyond traditional bucket-based clustering by using a Dual-Decoder Early Fusion Autoencoder to project student identities into a 16-dimensional joint latent space. The model fuses DistilBERT-derived semantic embeddings from student bios with structured tabular scores to encode identity, motivation, and collaborative potential in a single manifold.

## Technical File Map

```
backend/
├── data/
│   ├── synthetic_train_4000.csv      # Augmented training data (800 seeds x 5)
│   ├── synthetic_val_1000.csv        # Stranger Set: 200 unseen identities for generalization testing
│   └── standard_scaler.pkl           # Global Translation Key ensuring numeric consistency between training and inference
├── output/
│   ├── GroupGen_Encoder_Final_Safe.pt  # Production weights (0.0003 Val MSE)
│   └── final_report/
│       ├── robustness_metrics_summary.txt  # Monte Carlo tournament results table
│       └── model_performance_audit.json    # Deep learning benchmark summary
└── src/                               # Core model, clustering, and evaluation logic
```

### Why these files matter

- `backend/data/synthetic_train_4000.csv`: Provides the augmented training distribution used to fit the joint encoder while preserving representational diversity.
- `backend/data/synthetic_val_1000.csv`: Serves as the Stranger Set to validate true generalization on unseen student identities.
- `backend/data/standard_scaler.pkl`: Acts as the global scaler for consistent tabular normalization across training and inference stages.
- `backend/output/GroupGen_Encoder_Final_Safe.pt`: Represents the final, production-ready model state with validated 0.0003 validation MSE.
- `backend/output/final_report/`: Contains the assessment artifacts that support the research claims, including tournament metrics and benchmark summaries.

## Core Source Code

### Backend/src/ — Core Model & Evaluation Logic

| File | Purpose |
|------|---------|
| `__init__.py` | Package initialization for backend module imports |
| `api.py` | REST API layer for model serving and inference requests |
| `clustering.py` | Core clustering algorithms and feature vector computation; includes K-Medoids and feature scaling utilities |
| `data_loader.py` | CSV parsing and data validation; ensures student data format integrity |
| `evaluate_multimodal.py` | Comparative evaluation of clustering architectures (baseline vs. early fusion); generates performance reports |
| `evaluate_robustness.py` | Main evaluation script; runs the GroupGen-Encoder on validation set and computes tournament metrics (Silhouette, Davies-Bouldin, CH Index) |
| `generate_5000.py` | Synthetic data generation script for bootstrapping experimental datasets |
| `generate_groups.py` | Production pipeline; generates final student group assignments using trained encoder |
| `group_gen_intake.py` | User intake form handler for group generation parameters |
| `inspect_groups.py` | **Classroom audit utility** — Samples 30 random students, clusters them into 6 groups of 5, and prints demographic/semantic composition for manual verification of semantic manifold diversity |
| `kmedoids.py` | Custom K-Medoids (PAM) clustering implementation using Manhattan distance |
| `train_autoencoder.py` | Model training script; supports local and Colab environments; saves best weights to `backend/output/GroupGen_Encoder_Final_Safe.pt` |

### scripts/ — Standalone Entry Points

| File | Purpose |
|------|---------|
| `train_autoencoder.py` | **Primary entry point** for GroupGen-Encoder training; supports local and Colab environments; saves best weights to `backend/output/GroupGen_Encoder_Final_Safe.pt` |
| `generate_correlated_data.py` | Utility for generating synthetic student datasets with configurable correlation structures |

## Model Performance Benchmarks

- **Bottleneck dimension:** 16
- **Loss weighting:** 10:1 tabular-to-text reconstruction loss ratio
- **Text recovery:** 90%+ cosine similarity on reconstructed semantic embeddings

## The Semantic Manifold Defense

### Theoretical Discussion

Traditional baseline metrics such as Silhouette and Calinski-Harabasz favor rigid categorical splits because they reward exact matching in sparse discrete feature space. In contrast, GroupGen-Encoder prioritizes semantic nuance by embedding student identities in a continuous latent manifold that integrates both text-derived meaning and tabular skill signals.

Our model may exhibit lower Silhouette scores relative to bucket-based baselines, but it achieves superior Davies-Bouldin Index scores, which indicate tighter intra-cluster cohesion and clearer inter-cluster separation for socially consistent groups. This demonstrates that the encoder produces groups that are semantically aligned and more robust for collaborative student team formation.

## Setup & Usage

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB RAM minimum (16GB+ recommended for training)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd GroupGen
   ```

2. **Create and activate virtual environment (Windows):**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

   **macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Key dependencies:**
   - `torch>=2.0.0` — Deep learning framework
   - `transformers>=4.30.0` — DistilBERT tokenizer and models
   - `scikit-learn>=1.2.0` — StandardScaler, clustering utilities
   - `pandas>=1.5.0` — Data manipulation
   - `numpy>=1.23.0` — Numerical computing

### Training the Model

To train the GroupGen-Encoder from scratch:

```bash
python scripts/train_autoencoder.py
```

**Training parameters (configurable in script):**
- Epochs: 30
- Batch size: 100
- Learning rate: 0.005
- Optimizer: Adam
- Loss weighting: 10:1 (tabular:text)

**Output:** Saves best model to `backend/output/GroupGen_Encoder_Final_Safe.pt` when validation MSE improves.

### Evaluation

Run the final robustness evaluation on the validation set:

```bash
python backend/src/evaluate_robustness.py
```

**Expected outputs:**
- Console: Silhouette scores, Davies-Bouldin Index, Calinski-Harabasz Index
- File: `backend/output/final_report/robustness_metrics_summary.txt` (tournament results)
- File: `backend/output/final_report/model_performance_audit.json` (benchmark metrics)

### Generating Student Groups

To generate groups for a new cohort:

```bash
python backend/src/generate_groups.py
```

Provide a CSV file with student data in the format specified below.

### Classroom Audit (Semantic Manifold Verification)

To manually inspect how the GroupGen-Encoder mixes different student demographics and learning styles, run the classroom audit utility:

```bash
python backend/src/inspect_groups.py
```

This script:
- Randomly samples 30 students from the validation set (simulating one classroom)
- Clusters them into 6 groups of 5 using the trained encoder
- Prints a classroom audit showing:
  - Student ID, learning style, motivation, self-esteem, and work ethic for each group
  - First 50 characters of each student's bio (to verify semantic diversity)
  - Learning style entropy scores for each group and overall classroom (higher = more mixed learning styles)

**Expected output:** A formatted audit table demonstrating that the semantic manifold successfully creates diverse, cognitively balanced groups based on textual and behavioral embeddings.

### Input Data Format

Student CSV files must include:

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `Name` | String | Alice | Student identifier |
| `Gender` | String | Female | Male, Female, or Other |
| `Motivation` | Integer | 3 | 1 (Low) to 4 (High) |
| `Self_Esteem` | Integer | 2 | 1 (Low) to 4 (High) |
| `Work_Ethic` | Integer | 4 | 1 (Low) to 4 (High) |
| `Learning_Style` | String | Visual | Visual, Auditory, or Kinesthetic |
| `Text` | String | "I love..." | Student biography (50-150 words) |
| `Diversity` | String | Asian | Demographic category (optional) |

### Troubleshooting

**Issue: `ModuleNotFoundError: No module named 'torch'`**
- Solution: Ensure virtual environment is activated and run `pip install -r requirements.txt`

**Issue: CUDA out of memory during training**
- Solution: Reduce `batch_size` in `train_autoencoder.py` from 100 to 64 or 32

**Issue: Training script cannot find scaler or weights**
- Solution: Verify that `backend/data/standard_scaler.pkl` and training data exist before running training

**Issue: Evaluation script fails on inference**
- Solution: Ensure `backend/output/GroupGen_Encoder_Final_Safe.pt` is present; retrain if missing using `python scripts/train_autoencoder.py`


