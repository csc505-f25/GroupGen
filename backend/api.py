from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import numpy as np

# Import the logic we already verified in clustering.py
from clustering import (
    compute_feature_vector, 
    compute_distance_matrix, 
    enforce_group_size, 
    check_gender_isolation, 
    fix_gender_isolation, 
    check_diversity_isolation,
    fix_diversity_isolation
)
from kmedoids import kmedoids_pam

app = FastAPI(title="GroupGen API")

# Enable CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
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
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

    # 2. VALIDATE COLUMNS
    required_cols = ['Name', 'Motivation', 'Self_Esteem', 'Work_Ethic', 'Learning_Style', 'Gender', 'Diversity']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    # 3. RUN THE PIPELINE
    try:
        # A. Features
        feature_matrix = compute_feature_vector(df)
        euc_dist, dist_manhattan, _ = compute_distance_matrix(df, feature_matrix)

        # B. Clustering (K-Medoids Manhattan)
        n_students = len(df)
        n_groups = max(1, n_students // group_size)
        
        labels, _ = kmedoids_pam(dist_manhattan, n_groups, random_state=42)

        # C. Enforce Size
        labels = enforce_group_size(labels, group_size, feature_matrix=feature_matrix, metric='manhattan')

        # D. Apply Constraints (Locking)
        isolated_gender = check_gender_isolation(df, labels)
        if isolated_gender:
            labels = fix_gender_isolation(df, labels, euc_dist, isolated_gender)
        
        isolated_diversity = check_diversity_isolation(df, labels)
        if isolated_diversity:
            labels = fix_diversity_isolation(df, labels, euc_dist, isolated_diversity)

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
            "groups": response_groups
        }

    except Exception as e:
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Algorithm Error: {str(e)}")

if __name__ == "__api__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)