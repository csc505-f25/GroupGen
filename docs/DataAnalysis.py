from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# STEP 1: LOAD THE DATA
# =====================================================================
# Anchor paths to this script's folder so it runs from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = (
    SCRIPT_DIR
    / "Inclass_data"
    / "CSC412"
    / "CSC412 GroupGen Teaming Survey (Responses) - Form Responses 1.csv"
)
df = pd.read_csv(CSV_PATH)

# Google Forms exports often carry trailing/leading spaces in the question headers.
# Strip them so the rename map below matches reliably.
df.columns = df.columns.str.strip()

print(f"Loaded dataset with {df.shape[0]} responses and {df.shape[1]} columns.\n")

# =====================================================================
# STEP 2: RENAME COLUMNS (Using simple, understandable names)
# =====================================================================
column_mapping = {
    'Timestamp': 'survey_date',
    'Email Address': 'email_address',
    'What is your first and last name': 'student_name',
    
    # Learning Styles / Preferences
    'When I operate new equipment I generally:': 'learning_style_equipment',
    'When I cook a new dish, I like to': 'learning_style_cooking',
    'If I am teaching someone something new, I tend to': 'learning_style_teaching',
    
    # Confidence & Academic Self-Efficacy
    'I believe I will receive an excellent grade in this class.': 'confidence_expect_excellent_grade',
    "I'm certain I can understand the most difficult material presented in the readings for this course.": 'confidence_hard_readings',
    "I'm confident I can understand the basic concepts taught in this course.": 'confidence_basic_concepts',
    
    # Classroom Engagement
    'I sit near the front of the class if possible.': 'classroom_sit_at_front',
    'I am alert in classes': 'classroom_stay_alert',
    'I ask the instructor questions when clarification is needed.': 'classroom_ask_questions',
    
    # Study Habits
    'I arrive at classes and other meetings on time.': 'habit_punctual_to_class',
    'I devote sufficient study time to each of my courses.': 'habit_enough_study_time',
    'I schedule definite times and outline specific goals for my study time.': 'habit_set_study_goals',
    
    # Demographics
    'To which gender identity do you most identify?': 'gender_identity',
    'To which ethnicity do you most identify?': 'ethnicity_group'
}

df = df.rename(columns=column_mapping)

# Drop any accidental magic wand feedback columns out of numerical arrays
# Pandas converts duplicates to 'Column Name.1', 'Column Name.2' behind the scenes
magic_wand_cols = [c for c in df.columns if 'magic wand' in c.lower() or 'feedback' in c.lower()]
df = df.drop(columns=magic_wand_cols, errors='ignore')

# =====================================================================
# STEP 3: CONVERT & CLEAN TARGET SCALE METRICS
# =====================================================================
numeric_cols = [
    'confidence_expect_excellent_grade', 'confidence_hard_readings', 'confidence_basic_concepts',
    'classroom_sit_at_front', 'classroom_stay_alert', 'classroom_ask_questions',
    'habit_punctual_to_class', 'habit_enough_study_time', 'habit_set_study_goals'
]

# The classroom/habit questions are answered on a text frequency scale
# (Always/Usually/Sometimes/Never), so map them to a 5-point numeric scale.
# Includes the common "Somtimes" typo found in the export.
FREQUENCY_MAP = {
    'always': 5,
    'usually': 4,
    'sometimes': 3,
    'somtimes': 3,
    'rarely': 2,
    'never': 1,
}

# Ensure everything inside our numeric array handles string conversions elegantly
for col in numeric_cols:
    if col in df.columns:
        mapped = df[col].astype(str).str.strip().str.lower().map(FREQUENCY_MAP)
        # Confidence columns are already numeric (1-7); fall back to those values.
        df[col] = mapped.fillna(pd.to_numeric(df[col], errors='coerce'))

# Generate a unified composite column for confidence tracking
confidence_questions = [
    'confidence_expect_excellent_grade', 
    'confidence_hard_readings', 
    'confidence_basic_concepts'
]
df['overall_confidence_average'] = df[confidence_questions].mean(axis=1)

# =====================================================================
# STEP 4: PLOT AND EXPORT DEMOGRAPHIC PROFILE DISTRIBUTIONS
# =====================================================================
demo_cols = ['gender_identity', 'ethnicity_group']

for col in demo_cols:
    if col in df.columns:
        plt.figure(figsize=(8, 4))
        # Plot order sorted by density count
        sns.countplot(
            data=df, x=col, hue=col, palette='Set2', legend=False,
            order=df[col].value_counts().index,
        )
        plt.title(f"GroupGen Breakdown: {col.replace('_', ' ').title()}")
        plt.xlabel(col.replace('_', ' ').title())
        plt.ylabel('Student Counter')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'distribution_{col}.png')
        plt.close()

# =====================================================================
# STEP 5: COMPUTE EXTENSIVE DESCRIPTIVE STATISTICS
# =====================================================================
print("--- ALL CRITICAL METRICS FOR NUMERICAL COLUMNS ---")

metrics = ['min', 'max', 'mean', 'median', 'std', 'var']
detailed_stats = df[numeric_cols + ['overall_confidence_average']].agg(metrics).T
detailed_stats.columns = ['Min', 'Max', 'Average (Mean)', 'Median', 'StDev', 'Variance']

print(detailed_stats.round(3))
print("\n" + "="*60 + "\n")

# =====================================================================
# STEP 6: PLOT DISTRIBUTION CHART
# =====================================================================
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='overall_confidence_average', kde=True, color='skyblue', bins=8)
plt.title('Distribution of Overall Student Confidence')
plt.xlabel('Average Score Matrix')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'distribution_overall_confidence.png')
plt.close()

print("Mathematical metrics computed successfully!")
print(f"Distribution PNG charts saved to: {SCRIPT_DIR}")