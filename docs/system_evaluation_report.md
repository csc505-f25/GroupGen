# GroupGen: Formal System Evaluation Report

This report summarizes the rigorous technical audit of the **GroupGen** algorithmic architecture, covering data extraction, metric clustering, diversity scaling, and error bounds.

## 1. Data Ingestion & State Validation
**Score: Outstanding (10/10)**

The intake layer (`group_gen_intake.py`) employs a highly resilient "Fuzzy Logic" methodology for extracting CSV data:
- **Resilient Boundaries:** By utilizing the recurring "If you had a magic wand..." questions as array anchors, the script mathematically computes the integer span of the Self-Esteem, Motivation, and Work Ethic blocks. This makes the system "immune" to educators adding or removing survey questions. 
- **Defensive Type Mapping:** The system preempts standard `pandas` type coercion errors by intelligently stripping trailing strings via Regex, and mapping text representations of frequency (`"Always"`) and agreement (`"Strongly Agree"`) to discrete floats.
- **Data Safeguards:** If zero data is uploaded (or only "Ghost Rows" lacking names exist), the system halts execution before allocating mathematical buffers, gracefully throwing a `400 Bad Request` instead of triggering a server exception.

## 2. Algorithmic Choice (K-Medoids / PAM)
**Score: Excellent (9/10)**

The project employs a "Partitioning Around Medoids" (PAM) execution using **Manhattan Distance** (L1 Norm), which is the absolute gold standard for this specific dataset profile.
- Standard K-Means (Euclidean Distance) is highly sensitive to outliers and struggles geometrically with One-Hot Encoded variables (like the `Learning_Style` parameter). By running K-Medoids with Manhattan, the system computes the exact grid walk distance between Likert points and purely categorical bounds. 
- The features are clamped rigorously using `StandardScaler` ensuring that 1-7 Self-Esteem scales do not completely overwrite 1-4 Work Ethic variances structurally.

## 3. Constraint Mechanics (The "Locking" System)
**Score: Exceptional (10/10)**

Instead of using a naive "Least Happy" greedy algorithm—which violently ejects the most disgruntled student to a random donor group—GroupGen uses a **"Minimum-Cost Harmonic Target"** evaluation.
- When an isolation loop (Gender or Diversity) detects a flag, it isolates donors containing surplus/matching identities.
- It then evaluates the matrix distance of *every* applicable candidate specifically against the *Receiver's* Medoid center point.
- **The Result:** The algorithm selects the "Most Harmonious Recipient" (the exact student who fits perfectly into the new group without damaging its internal skill variances) ensuring demographic constraints are fixed with the absolute minimum disruption to mathematical entropy!

## 4. Evaluation Matrix (`evaluate_clustering.py`)
**Score: Great (8.5/10)**

The research layer handles algorithm performance mathematically:
- Computes `Silhouette Scores` globally, dynamically adjusting dependent on metric structure (Euclidean vs Manhattan vs Gower via predefined pairwise calculations). 
- Extracts deep "Mean Gender Entropy" and specific variance per-cluster statistics, allowing analysts to accurately measure how effective the base unsupervised clustering was prior to invoking the forced demographic constraints.

## Conclusion

The architecture of GroupGen stands as a mathematically sound, highly defensible student distribution application. With robust dynamic parsing, the adoption of L1 norm metrics over basic KMeans, and a highly innovative constraint lock that prevents systemic isolation, the platform is robustly structured to execute at scale.
