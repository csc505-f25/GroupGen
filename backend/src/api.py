from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import numpy as np
import os
import datetime
import time

# --- Logging Helper ---
def log_pipeline_step(step_name, data=None, msg=""):
    """
    Safely logs pipeline steps to 'output/pipeline_log.txt'.
    Does not crash on objects that can't be printed.
    """
    try:
        # Ensure output dir exists
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "pipeline_log.txt")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"STEP: {step_name}  [{timestamp}]\n")
            f.write(f"{'='*60}\n")
            
            if msg:
                f.write(f"Message: {msg}\n")
            
            if data is not None:
                # Handle Pandas DataFrame
                if isinstance(data, pd.DataFrame):
                    f.write(f"DataFrame Shape: {data.shape}\n")
                    f.write("Columns: " + ", ".join(data.columns.tolist()) + "\n")
                    f.write("First 5 Rows:\n")
                    f.write(data.head(5).to_string() + "\n")
                    
                    # Log stats if numeric columns exist
                    try:
                        f.write("\nQuick Stats:\n")
                        f.write(data.describe().to_string() + "\n")
                    except:
                        pass
                        
                # Handle Numpy Arrays (Matrices)
                elif isinstance(data, np.ndarray):
                    f.write(f"Array Shape: {data.shape}\n")
                    f.write("First 5 Rows (or elements):\n")
                    if data.ndim == 1:
                         f.write(str(data[:10]) + " ...\n")
                    else:
                         f.write(str(data[:5]) + "\n")
                         
                # Handle Dicts/Lists
                elif isinstance(data, (dict, list)):
                     import json
                     # Try nice print, valid json fallback
                     try:
                         f.write(json.dumps(data, indent=2, default=str) + "\n")
                     except:
                         f.write(str(data) + "\n")
                
                else:
                    f.write(str(data) + "\n")
                    
            f.write("\n")
            
    except Exception as e:
        print(f"Logging failed (but pipeline continues): {e}")

# Import the logic we already verified in clustering.py
from .clustering import (
    compute_feature_vector, 
    compute_distance_matrix, 
    enforce_group_size, 
    check_gender_isolation, 
    fix_gender_isolation, 
    check_diversity_isolation,
    fix_diversity_isolation
)
from .kmedoids import kmedoids_pam
from .group_gen_intake import process_google_form

app = FastAPI(title="GroupGen API")

# Enable CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "GroupGen API is running"}

@app.post("/generate-groups")
async def generate_groups(
    file: UploadFile = File(...), 
    group_size: int = 5,
):
    """
    Stateless Endpoint: Receives CSV -> Returns Groups + Stats
    """
    # 1. READ FILE
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        contents = await file.read()
        df = process_google_form(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV is entirely empty or only contains invalid 'Ghost Rows'. Please upload a valid CSV with student data.")


    # 2. VALIDATE COLUMNS
    required_cols = ['Name', 'Motivation', 'Self_Esteem', 'Work_Ethic', 'Learning_Style', 'Gender', 'Diversity']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    # 3. RUN THE PIPELINE
    try:
        # A. Features
        log_pipeline_step("1. INGESTION", df, "Loaded CSV and cleaned data.")
        
        feature_matrix = compute_feature_vector(df)
        log_pipeline_step("2. VECTORIZATION", feature_matrix, "Converted students to numerical vectors.")
        
        euc_dist, dist_manhattan, _ = compute_distance_matrix(df, feature_matrix)
        log_pipeline_step("3. DISTANCE CALCULATION", dist_manhattan, "Calculated Manhattan distance between all students.")

        # B. Clustering (K-Medoids Manhattan)
        n_students = len(df)
        n_groups = max(1, int(np.ceil(n_students / group_size)))
        
        # Prevent individual isolation (groups of 1) by reducing group count
        # until the minimum mathematical group size is at least 2.
        while n_groups > 1 and n_students // n_groups < 2:
            n_groups -= 1
        
        labels, _ = kmedoids_pam(dist_manhattan, n_groups, random_state=42)
        log_pipeline_step("4. INITIAL CLUSTERING (K-Medoids)", labels, f"Created {n_groups} initial clusters.")

        # C. Enforce Size
        labels = enforce_group_size(labels, group_size, feature_matrix=feature_matrix, metric='manhattan', expected_n_clusters=n_groups)
        log_pipeline_step("5. BALANCING (Enforce Size)", labels, f"Balanced groups to approx size {group_size}.")

        # D. Apply Constraints (Locking)
        isolated_gender = check_gender_isolation(df, labels)
        if isolated_gender:
            labels = fix_gender_isolation(df, labels, dist_manhattan, isolated_gender)
            log_pipeline_step("6. GENDER CONSTRAINT CHECK", labels, "Fixed gender isolation.")
        
        isolated_diversity = check_diversity_isolation(df, labels)
        if isolated_diversity:
            labels = fix_diversity_isolation(df, labels, dist_manhattan, isolated_diversity)
            log_pipeline_step("7. DIVERSITY CONSTRAINT CHECK", labels, "Fixed diversity isolation.")

        # 4. FORMAT RESPONSE & CALCULATE STATS
        df['Group_ID'] = labels + 1
        
        response_groups = []
        unique_ids = sorted(df['Group_ID'].unique())
        
        for g_id in unique_ids:
            # Get members as a list of dicts (handle NaN)
            group_df = df[df['Group_ID'] == g_id]
            members = group_df.replace({np.nan: None}).to_dict(orient='records')
            
            # --- THE COOL STATS LOGIC YOU WANTED ---
            stats = {
                "size": len(members),
                "avg_motivation": round(group_df['Motivation'].mean(), 2),
                "avg_work_ethic": round(group_df['Work_Ethic'].mean(), 2),
                "gender_balance": group_df['Gender'].value_counts().to_dict(),
                "learning_styles": group_df['Learning_Style'].mode().tolist() # Most common style
            }
            
            response_groups.append({
                "id": int(g_id),
                "members": members,
                "stats": stats
            })

        return {
            "status": "success",
            "total_students": n_students,
            "total_groups": len(unique_ids),
            "groups": response_groups,
        }
        

    except Exception as e:
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Algorithm Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)