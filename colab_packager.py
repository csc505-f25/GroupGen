import os
import shutil
import zipfile

def create_colab_package():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    upload_dir = os.path.join(base_dir, 'colab_upload')
    
    # Files to include
    files_to_copy = [
        os.path.join(base_dir, 'backend', 'deep_learning', 'scripts', 'train_autoencoder.py'),
        os.path.join(base_dir, 'backend', 'deep_learning', 'models', 'multimodal_autoencoder.py'),
        os.path.join(base_dir, 'backend', 'data', 'synthetic_multimodal_5000.csv')
    ]
    
    # Create or clean the colab_upload directory
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    os.makedirs(upload_dir)
    
    print(f"Creating Colab package in: {upload_dir}")
    
    # Copy files
    for file_path in files_to_copy:
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(upload_dir, file_name)
            shutil.copy2(file_path, dest_path)
            print(f"  [+] Copied: {file_name}")
        else:
            print(f"  [-] WARNING: File not found - {file_path}")
            
    # Create the ZIP archive
    zip_path = os.path.join(base_dir, 'colab_upload.zip')
    print(f"\nZipping files to: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(upload_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to zip, arcname ensures flat structure in zip
                zipf.write(file_path, arcname=file)
                print(f"  [+] Zipped: {file}")
                
    print("\nSUCCESS: Colab Package successfully created!")
    print("\n" + "="*60)
    print("COLAB BOILERPLATE INSTRUCTIONS")
    print("="*60)
    print("1. Upload 'colab_upload.zip' to the root of your Google Colab instance.")
    print("2. Create a new code cell at the very top of your notebook and run:\n")
    print("!unzip -o colab_upload.zip")
    print("!pip install torch transformers pandas numpy scikit-learn matplotlib tqdm")
    print("\n3. In the next cell, you can run your training script:\n")
    print("!python train_autoencoder.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    create_colab_package()
