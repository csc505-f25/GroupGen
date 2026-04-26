import pandas as pd
import numpy as np
import os
import shutil

def generate_golden_dataset(n_samples=1000):
    np.random.seed(42)
    
    # 4 Archetypes:
    archetypes = np.random.choice([0, 1, 2, 3], size=n_samples)
    data = {'Name': [f"Student_{i}" for i in range(n_samples)]}
    
    motivation, self_esteem, work_ethic, text, ls, gender, div = [], [], [], [], [], [], []
    
    text_pools = [
        ["I am extremely motivated to score a perfect 100 on this project.", "I thrive on hard work and I expect my team to perform flawlessly.", "I am highly ambitious and love taking on difficult problems."],
        ["I really don't care about this class, just trying to pass.", "I struggle significantly and have zero motivation to work.", "This material is way too hard and I'm not interested in trying."],
        ["I'm brilliant so I don't really need to study much.", "Why work hard when my natural talent carries me?", "I'm very confident I'll pass even without much effort."],
        ["I work constantly because I'm terrified of failing.", "I suffer from anxiety and compensate by over-preparing.", "I'm not confident in my skills so I just work extreme overtime."]
    ]
    
    learning_styles = ['Visual', 'Auditory', 'Kinesthetic']
    genders = ['Male', 'Female', 'Other']
    diversities = ['Category_A', 'Category_B', 'Category_C']
    
    for a in archetypes:
        # Base scores
        if a == 0:
            m_base, s_base, w_base = np.random.choice([3, 4]), np.random.choice([3, 4]), np.random.choice([3, 4])
        elif a == 1:
            m_base, s_base, w_base = np.random.choice([1, 2]), np.random.choice([1, 2]), np.random.choice([1, 2])
        elif a == 2:
            m_base, s_base, w_base = np.random.choice([1, 2]), np.random.choice([3, 4]), np.random.choice([1, 2])
        elif a == 3:
            m_base, s_base, w_base = np.random.choice([3, 4]), np.random.choice([1, 2]), np.random.choice([3, 4])
            
        # Inject standard variance (+/- 1 occasionally) so the coordinates form messy "clouds" rather than tight dots
        def add_noise(val):
            return int(np.clip(val + np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]), 1, 4))
            
        motivation.append(add_noise(m_base))
        self_esteem.append(add_noise(s_base))
        work_ethic.append(add_noise(w_base))
            
        # Choose their Learning Style first so we can weave it into their text
        chosen_ls = np.random.choice(learning_styles)
        ls.append(chosen_ls)
        
        ls_phrases = {
            'Visual': [" I learn best through diagrams and visual charts.", " Seeing the material visually really helps me.", " I am a very visual oriented learner."],
            'Auditory': [" I prefer listening to discussions and lectures.", " Hearing concepts explained out loud works best for me.", " I'm definitely an auditory learner."],
            'Kinesthetic': [" I need to do hands-on exercises to understand.", " Learning by directly doing experiments is my approach.", " I am a highly kinesthetic, tactile learner."]
        }
        
        # 20% of the time, inject active noise (saying something that contradicts their actual spreadsheet data)
        if np.random.rand() < 0.20:
            all_texts = [text for pool in text_pools for text in pool]
            base_text = np.random.choice(all_texts)
            # Give them a random, mismatched learning style phrase as noise
            mismatched_ls = np.random.choice(learning_styles)
            ls_phrase = np.random.choice(ls_phrases[mismatched_ls])
        else:
            base_text = np.random.choice(text_pools[a])
            ls_phrase = np.random.choice(ls_phrases[chosen_ls])
            
        final_text = f"{base_text}{ls_phrase}"
        text.append(final_text)
        
        gender.append(np.random.choice(genders, p=[0.45, 0.45, 0.1]))
        div.append(np.random.choice(diversities))
        
    data['Motivation'] = motivation
    data['Self_Esteem'] = self_esteem
    data['Work_Ethic'] = work_ethic
    data['Learning_Style'] = ls
    data['Gender'] = gender
    data['Diversity'] = div
    data['Text'] = text
    
    df = pd.DataFrame(data)
    
    # Path logic ensuring that wherever from the root this is executed, it routes to backend/data correctly
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'backend', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    main_file = os.path.join(data_dir, 'synthetic_multimodal_1000.csv')
    df.to_csv(main_file, index=False)
    
    # Generate required slices for final paper evaluation lengths
    for n in [30, 50, 100]:
        slice_path = os.path.join(data_dir, f'synthetic_multimodal_{n}.csv')
        df.head(n).to_csv(slice_path, index=False)
        print(f"Generated evaluation subset: synthetic_multimodal_{n}.csv")
    
    # We clone it to 5000 strictly for baseline backwards compatibility in case it's hardcoded anywhere else
    shutil.copy(main_file, os.path.join(data_dir, 'synthetic_multimodal_5000.csv'))
    print(f"Golden dataset generation fully completed inside {data_dir}!")

if __name__ == "__main__":
    generate_golden_dataset()
