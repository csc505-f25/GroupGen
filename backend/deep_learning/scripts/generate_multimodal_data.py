"""
Generates 5000 synthetic student profiles including Free-Response text
so we can train the Multimodal Autoencoder locally without API keys.
"""

import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_multimodal(n_samples=5000):
    np.random.seed(42)
    random.seed(42)

    names = [f"Student_{i}" for i in range(n_samples)]
    genders = np.random.choice(['Male', 'Female', 'Non-Binary'], n_samples, p=[0.48, 0.48, 0.04])
    diversities = np.random.choice(
        ['Caucasian', 'Asian', 'Hispanic/LatinX', 'Black/African', 'Native American/ Pacific Islander'], 
        n_samples
    )
    learning_styles = np.random.choice(['Visual', 'Auditory', 'Kinesthetic'], n_samples)
    
    # Generate scores with some bell curve logic
    motivation = np.clip(np.round(np.random.normal(3, 1, n_samples)), 1, 4).astype(int)
    self_esteem = np.clip(np.round(np.random.normal(3, 1, n_samples)), 1, 4).astype(int)
    work_ethic = np.clip(np.round(np.random.normal(3, 1, n_samples)), 1, 4).astype(int)

    # ---------------------------------------------------------
    # Text Correlation Dictionaries
    # ---------------------------------------------------------
    mot_high = [
        "I am very driven and usually take the lead on making sure we hit all the rubric points.",
        "I love diving deep into the material and organizing the progression of our work.",
        "I always aim for an A+ and will happily manage the team's schedule."
    ]
    mot_low = [
        "I prefer just doing whatever part is assigned to me so I can clock out.",
        "I'm not looking to stress over this, just want to get it done.",
        "Honestly I struggle to stay engaged unless someone tells me exactly what to do."
    ]

    we_high = [
        "I usually finish my code way before the deadline.",
        "I'm extremely punctual and expect my teammates to also contribute their fair share on time.",
        "I will happily spend the weekend polishing our project until it is perfect."
    ]
    we_low = [
        "I have a lot of other classes so this isn't my main priority.",
        "I tend to leave things to the last minute but it usually works out.",
        "I struggle with deadlines sometimes."
    ]

    style_dict = {
        'Visual': ["I need to see charts and diagrams to understand the architecture.", "I prefer writing on whiteboards to plan."],
        'Auditory': ["I work best when we talk through the architecture out loud.", "I like bouncing ideas off people verbally."],
        'Kinesthetic': ["I learn by doing, so just let me start coding the MVP.", "I don't like reading manuals, I just want to start building hands-on."]
    }

    # Generate the text paragraphs
    texts = []
    for m, w, l in zip(motivation, work_ethic, learning_styles):
        text_parts = []
        
        # Motivation logic
        if m >= 3: text_parts.append(random.choice(mot_high))
        elif m <= 2: text_parts.append(random.choice(mot_low))
            
        # Work ethic logic
        if w >= 3: text_parts.append(random.choice(we_high))
        elif w <= 2: text_parts.append(random.choice(we_low))
            
        # Learning style logic
        text_parts.append(random.choice(style_dict[l]))
        
        random.shuffle(text_parts) # Shuffle so it reads loosely like a paragraph
        texts.append(" ".join(text_parts))

    # Compile DataFrame
    df = pd.DataFrame({
        'Name': names,
        'Gender': genders,
        'Diversity': diversities,
        'Learning_Style': learning_styles,
        'Motivation': motivation,
        'Self_Esteem': self_esteem,
        'Work_Ethic': work_ethic,
        'Text': texts
    })

    # Save to data directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'synthetic_multimodal_5000.csv')
    df.to_csv(out_path, index=False)
    print(f"Generated {n_samples} multimodal synthetic profiles at {out_path}!")

if __name__ == "__main__":
    generate_synthetic_multimodal()
