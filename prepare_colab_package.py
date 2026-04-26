"""
Colab Package Preparation Script
Prepares the GroupGen training code for Google Colab by bundling necessary files.
Run this script locally to generate a colab_upload.zip file ready for Colab.
"""

import os
import shutil
import zipfile
from pathlib import Path

def prepare_colab_package():
    """
    Creates a colab_upload folder with all necessary files and packages it as ZIP.
    """
    
    # Define base directory
    script_dir = Path(__file__).resolve().parent
    backend_src = script_dir / 'backend' / 'src'
    backend_models = script_dir / 'backend' / 'deep_learning' / 'models'
    backend_data = script_dir / 'backend' / 'data'
    
    # Create colab_upload directory
    colab_upload_dir = script_dir / 'colab_upload'
    if colab_upload_dir.exists():
        print(f"Removing existing {colab_upload_dir}...")
        shutil.rmtree(colab_upload_dir)
    
    colab_upload_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {colab_upload_dir}")
    
    # 1. Copy training script (refactored for Colab)
    train_script = backend_src / 'train_autoencoder.py'
    if train_script.exists():
        shutil.copy(train_script, colab_upload_dir / 'train_autoencoder.py')
        print(f"✓ Copied: train_autoencoder.py")
    else:
        print(f"⚠ WARNING: {train_script} not found!")
    
    # 2. Copy multimodal autoencoder model
    model_file = backend_models / 'multimodal_autoencoder.py'
    if model_file.exists():
        shutil.copy(model_file, colab_upload_dir / 'multimodal_autoencoder.py')
        print(f"✓ Copied: multimodal_autoencoder.py")
    else:
        print(f"⚠ WARNING: {model_file} not found!")
    
    # 3. Copy dataset (1000-student golden set)
    dataset_file = backend_data / 'synthetic_multimodal_1000.csv'
    if dataset_file.exists():
        shutil.copy(dataset_file, colab_upload_dir / 'synthetic_multimodal_1000.csv')
        print(f"✓ Copied: synthetic_multimodal_1000.csv")
    else:
        print(f"⚠ WARNING: {dataset_file} not found!")
    
    # 4. Create README for Colab
    readme_content = """# GroupGen Multimodal Autoencoder - Colab Ready Package

## Setup Instructions

### 1. Upload to Colab
- Upload the `colab_upload.zip` file to your Google Colab session
- Unzip it in the first cell

### 2. First Cell: Install Dependencies & Unzip
```python
!pip install --upgrade torch transformers pandas numpy scikit-learn gower
!unzip -q colab_upload.zip
%cd colab_upload
```

### 3. Second Cell: Run Training
```python
!python train_autoencoder.py
```

## File Structure
- `train_autoencoder.py` - Main training script (Colab-ready)
- `multimodal_autoencoder.py` - Model definition (the "Butterfly" model)
- `synthetic_multimodal_1000.csv` - Training dataset (1000 students)

## What to Expect
- **Training Time**: ~2-5 minutes on GPU
- **Output**: `multimodal_autoencoder_final.pt` - Model weights
- **Output**: `standard_scaler.pkl` - Feature scaler
- **Logs**: Epoch-wise Tabular MSE, Text MSE, and Weighted Loss
- **Evaluation**: Text Recovery Cosine Similarity metric

## Key Features
✅ Automatic GPU detection (CUDA in Colab)
✅ Weighted multimodal loss (10x tabular, 1x text)
✅ Balanced learning across modalities
✅ Path-agnostic (works in any directory)
✅ Production-ready weights saved

## Paper Methodology Notes
The training logs provide:
- **Tabular MSE**: Reconstruction error on numerical features
- **Text MSE**: Reconstruction error on 768-D DistilBERT embeddings
- **Weighted Loss**: Combined loss with 10:1 weighting
- **Cosine Similarity**: Text fidelity metric after training

All metrics are logged per epoch for reproducibility in your paper.
"""
    
    with open(colab_upload_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    print(f"✓ Created: README.md (Colab setup instructions)")
    
    # 5. Create Colab boilerplate notebook snippet
    boilerplate_content = """# ============================================================
# GOOGLE COLAB SETUP - Run this first!
# ============================================================

# Cell 1: Install & Setup
!pip install --upgrade torch transformers pandas numpy scikit-learn gower scipy matplotlib
!unzip -q colab_upload.zip
%cd colab_upload
!ls -la

# ============================================================
# Cell 2: Train the Model
# ============================================================
!python train_autoencoder.py

# ============================================================
# Cell 3 (Optional): Download Results
# ============================================================
from google.colab import files
files.download('multimodal_autoencoder_final.pt')
files.download('standard_scaler.pkl')
"""
    
    with open(colab_upload_dir / 'COLAB_BOILERPLATE.txt', 'w') as f:
        f.write(boilerplate_content)
    print(f"✓ Created: COLAB_BOILERPLATE.txt (copy-paste for Colab)")
    
    # 6. Create ZIP file
    zip_path = script_dir / 'colab_upload.zip'
    if zip_path.exists():
        zip_path.unlink()
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in colab_upload_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(colab_upload_dir.parent)
                zipf.write(file_path, arcname)
    
    print(f"\n✓ Created: {zip_path}")
    print(f"  File size: {zip_path.stat().st_size / (1024*1024):.2f} MB")
    
    print("\n" + "="*60)
    print("✅ COLAB PACKAGE READY!")
    print("="*60)
    print(f"\nLocation: {zip_path}")
    print(f"\nNext Steps:")
    print(f"1. Download colab_upload.zip from your file explorer")
    print(f"2. Go to Google Colab: https://colab.research.google.com")
    print(f"3. Create a new notebook and upload colab_upload.zip")
    print(f"4. Copy-paste the commands from COLAB_BOILERPLATE.txt")
    print(f"5. Run cells in order")
    print("\n" + "="*60)

if __name__ == "__main__":
    prepare_colab_package()
