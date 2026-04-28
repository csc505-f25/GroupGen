import pandas as pd
import numpy as np
import random
import os

def generate_leakage_safe_data(n_seeds=1000):
    np.random.seed(42)
    random.seed(42)

    # 1. GENERATE THE 1000 UNIQUE SEED STUDENTS
    # ---------------------------------------------------------
    def create_profiles(n):
        learning_styles = np.random.choice(['Visual', 'Auditory', 'Kinesthetic'], n)
        motivation = np.clip(np.round(np.random.normal(3, 1, n)), 1, 4).astype(int)
        self_esteem = np.clip(np.round(np.random.normal(3, 1, n)), 1, 4).astype(int)
        work_ethic = np.clip(np.round(np.random.normal(3, 1, n)), 1, 4).astype(int)
        
        # Text Snippets (Same as your original logic)
        mot_high = ["I am very driven...", "I love diving deep...", "I always aim for an A+..."]
        mot_low = ["I prefer just doing...", "I'm not looking to stress...", "Honestly I struggle..."]
        we_high = ["I usually finish my code...", "I'm extremely punctual...", "I will happily spend..."]
        we_low = ["I have a lot of other classes...", "I tend to leave things...", "I struggle with deadlines..."]
        style_dict = {
            'Visual': ["I need to see charts...", "I prefer writing on whiteboards..."],
            'Auditory': ["I work best when we talk...", "I like bouncing ideas..."],
            'Kinesthetic': ["I learn by doing...", "I don't like reading manuals..."]
        }

        texts = []
        for m, w, l in zip(motivation, work_ethic, learning_styles):
            parts = []
            if m >= 3: parts.append(random.choice(mot_high))
            elif m <= 2: parts.append(random.choice(mot_low))
            if w >= 3: parts.append(random.choice(we_high))
            elif w <= 2: parts.append(random.choice(we_low))
            parts.append(random.choice(style_dict[l]))
            random.shuffle(parts)
            texts.append(" ".join(parts))
            
        return pd.DataFrame({
            'Learning_Style': learning_styles, 'Motivation': motivation,
            'Self_Esteem': self_esteem, 'Work_Ethic': work_ethic, 'Text': texts
        })

    seeds_df = create_profiles(n_seeds)

    # 2. PERFORM THE 80/20 SPLIT AT THE SEED LEVEL
    # ---------------------------------------------------------
    seeds_df = seeds_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_seeds = seeds_df.iloc[:800].copy()
    val_seeds = seeds_df.iloc[800:].copy()

    # 3. AUGMENT THE TRAINING SEEDS (5x to get 4000)
    # ---------------------------------------------------------
    train_augmented = []
    for _ in range(5):
        temp = train_seeds.copy()
        # Add slight Gaussian noise to numeric values to create 'clones'
        for col in ['Motivation', 'Self_Esteem', 'Work_Ethic']:
            noise = np.random.normal(0, 0.1, size=len(temp))
            temp[col] = np.clip(np.round(temp[col] + noise), 1, 4).astype(int)
        train_augmented.append(temp)
    
    train_final = pd.concat(train_augmented).sample(frac=1).reset_index(drop=True)

    # 4. SAVE THE SEPARATE FILES
    # ---------------------------------------------------------
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    train_final.to_csv(os.path.join(output_dir, 'synthetic_train_4000.csv'), index=False)
    val_seeds.to_csv(os.path.join(output_dir, 'synthetic_val_1000.csv'), index=False)

    print("✅ Successfully generated leakage-safe datasets!")
    print(f"Training: 4000 students (800 seeds augmented 5x)")
    print(f"Validation: 200 students (Unique individuals never seen by model)")

if __name__ == "__main__":
    generate_leakage_safe_data()