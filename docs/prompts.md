# GroupGen: Prompt Library & AI Orchestration

This library provides optimized prompts for interacting with the GroupGen system. These are designed for use with advanced LLMs (Gemini 1.5 Pro, GPT-4, Claude 3) to refine backend logic, generate communications, or simulate datasets.

---

## 0. Global System Context (The "System Persona")
*Always prepend this to complex technical requests to ensure the AI understands the GroupGen architecture.*

```text
You are an expert Data Scientist and Educational Technologist specializing in algorithmic fairness. You are working on "GroupGen," a Python-based student grouping system. 

Key Architectural Details:
- Backend: Python (Pandas, Scikit-learn, Gower).
- Algorithm: K-Medoids (PAM) with Manhattan distance (optimized for mixed-type data).
- Core Feature: A "Locking Mechanism" (post-clustering swap logic) that prevents demographic isolation (solo-status) of minority groups.
- Metrics: Silhouette Score (cohesion), Gender/Diversity Entropy (balance), and Skill Variance.
```

---

## 1. Technical Development & Optimization
*Use these to tune the engine in `backend/clustering.py` and `backend/kmedoids.py`.*

### A. Tuning Distance Weights
> "Analyze the current distance matrix calculation. We need to prioritize 'Learning_Style' diversity without sacrificing 'Motivation' balance. Propose a `weighted_distance` function that applies a 1.5x multiplier to categorical mismatches in 'Learning_Style'. How should we normalize the numerical 'Motivation' (1-4) and 'Work_Ethic' (1-4) scores to ensure they aren't overshadowed by the weighted categorical variables?"

### B. Optimizing the "Locking Mechanism"
> "The current `fix_gender_isolation` function uses a greedy swap approach. Suggest a more computationally efficient heuristic (e.g., Simulated Annealing or a Min-Cost Flow approach) to resolve isolation constraints while minimizing the increase in the total Within-Cluster Sum of Squares (WCSS). Provide a Python pseudocode implementation."

---

## 2. Pedagogical & Stakeholder Communications
*Generate human-centric output based on `final_groups_report.txt`.*

### A. The "Team Launch" Email (Student-Facing)
> "Act as a supportive university professor. Using the following group data [Insert Data], draft a personalized 'Team Launch' email for 'Group 7'. 
> 1. Highlight their complementary 'Learning Styles' (e.g., pairing Kinesthetic with Visual).
> 2. Frame the 'Motivation' balance as a way to ensure shared leadership.
> 3. Use a tone that is encouraging, professional, and emphasizes psychological safety."

### B. Methodology Statement (Syllabus/Institutional)
> "Write a 250-word 'Statement of Algorithmic Fairness' for a university provost. Explain how GroupGen moves beyond random assignment to actively prevent 'solo-status' isolation for minority students. Explicitly mention the K-Medoids approach and how it preserves the integrity of skill-based clusters while prioritizing inclusivity."

---

## 3. QA & Synthetic Data Generation
*Generate edge-case datasets to stress-test the pipeline.*

### A. The "Worst-Case Scenario" Dataset
> "Generate a 50-row CSV following the GroupGen schema (Name, Gender, Motivation, Self_Esteem, Work_Ethic, Learning_Style, Diversity). Create a 'High-Constraint' scenario where:
> 1. Gender is skewed 85% Male / 15% Female.
> 2. Diversity categories are fragmented (e.g., 5 students across 4 different minority categories).
> 3. Motivation scores are polarized (only 1s and 4s).
> This will be used to test the limits of the 'Smart Fill' and 'Isolation Fix' logic."

---

## 4. Strategic Vision & Future Features
*For Phase 3 (Integration) and Phase 4 (Scaling).*

### A. Sentiment-Driven Grouping
> "We want to incorporate student 'Self-Reflections' into the grouping logic. Design a workflow using a `transformers` pipeline to extract a 'Collaborative Sentiment' score from text strings. How should this new feature be integrated into the existing `data_loader.py` and `clustering.py` without breaking the Gower distance calculation?"

### B. API Design (FastAPI)
> "Propose a REST API structure for GroupGen using FastAPI. Include endpoints for `/upload-csv`, `/generate-groups`, and `/export-report`. How should we handle the asynchronous nature of the clustering process for large datasets (n > 1000)?"

---

## 5. Debugging & Metric Interpretation
*Use these when the pipeline results are suboptimal.*

### A. Silhouette Score Analysis
> "My current run yielded a Silhouette Score of 0.12, which is quite low. Based on the GroupGen architecture, diagnose three potential reasons for this (e.g., high feature dimensionality, overlapping categorical values, or inappropriate group size). Suggest specific code changes to `backend/clustering.py` to improve cohesion."

### B. Swap Logic Failure
> "The 'Locking Mechanism' failed to resolve gender isolation in 2 out of 6 groups. Analyze the `backend/output/pipeline_log.txt` [Paste Log] and explain why no valid swap candidates were found. Is this a data density issue or a logical constraint in the swapping heuristic?"

---

## Best Practices for Using This Library
1. **Provide Context**: Always paste the relevant snippet of code or the `final_groups_report.txt` after the prompt.
2. **Specify Output Format**: If you need code, ask for "Pythonic, documented code." If you need an email, ask for "Markdown formatting."
3. **Iterate**: If the AI's first suggestion is too complex, follow up with: "Simplify this implementation for a class of 30 students."
