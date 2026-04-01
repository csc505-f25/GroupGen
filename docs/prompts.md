# GroupGen: The Prompt Engineering Library

This library is a collection of high-fidelity prompts designed to orchestrate the **GroupGen** ecosystem. Whether you are tuning the K-Medoids algorithm, drafting educator reports, or expanding the Next.js frontend, use these templates to get the most out of LLMs like Gemini 1.5 Pro, GPT-4, or Claude 3.5.

---

## 🏗️ 0. Global System Context (The "System Persona")
*Copy and paste this at the start of any new chat session to provide the AI with full architectural awareness.*

> **Persona:** You are a Senior Fullstack Engineer and Data Scientist specializing in educational technology and algorithmic fairness.
>
> **Project Overview:** "GroupGen" is an automated student grouping system that balances academic traits with demographic inclusivity.
> - **Backend:** Python, FastAPI, Pandas, Scikit-learn.
> - **Clustering:** K-Medoids (PAM) using Manhattan/Gower distance to handle mixed-type data (Numerical: Motivation, Work Ethic; Categorical: Learning Style, Gender, Diversity).
> - **Key Innovation:** A post-clustering "Locking Mechanism" that swaps students to prevent demographic isolation (solo-status) while preserving cluster integrity.
> - **Frontend:** Next.js 14+, TypeScript, Tailwind CSS, providing a dashboard for CSV uploads and group visualization.

---

## 🧬 1. Backend & Algorithmic Optimization
*Focus: Logic refinement, distance metrics, and performance tuning.*

### A. Implementing Weighted Gower Distance
> "We need to adjust the clustering to weigh 'Learning_Style' more heavily than other traits. Currently, we use a standard Manhattan distance on one-hot encoded variables. Propose a custom distance function that:
> 1. Applies a 2.0x weight to mismatches in 'Learning_Style'.
> 2. Normalizes numerical 'Motivation' scores to a 0-1 range before distance calculation.
> 3. Integrates seamlessly with the existing `kmedoids_pam` implementation in `backend/kmedoids.py`.
> Explain how this will affect the Silhouette Score vs. the cluster interpretability."

### B. Heuristic Swap Logic (Simulated Annealing)
> "The current 'Locking Mechanism' in `backend/clustering.py` uses a greedy approach to fix solo-status isolation. Design a more robust heuristic using **Simulated Annealing**. The energy function should minimize both (a) demographic isolation and (b) the increase in Within-Cluster Sum of Squares (WCSS). Provide a Python implementation that handles edge cases where no valid swaps are available."

### C. Dynamic Form Intake & Likert Parsing
> "We need to dynamically parse a raw Google Form CSV. Review `backend/group_gen_intake.py`. 
> 1. Identify repeated 'Magic Wand' questions to act as unbreakable column boundary anchors. Use these indexes to slice the dataframe into sections (Self-Esteem, Motivation, Work Ethic) so the script is immune to added or deleted questions.
> 2. Implement a robust `_parse_survey_score` helper that intercepts text strings. It should extract leading digits first, and cleanly map frequency words like 'Always' (4) or 'Sometimes' (2) to numeric floats before pandas coercion.
> Explain how this dynamic slicing prevents index-shifting errors and NaN data corruption."

---

## 🎨 2. Frontend & UI/UX Orchestration
*Focus: Dashboard enhancements, data visualization, and user experience.*

### A. Dynamic Group Comparison Component
> "Review the current `page.tsx` results grid. I want to add a 'Comparative Analytics' drawer that appears when two groups are selected. 
> 1. Use **Recharts** or a similar library to show a radar chart comparing the average 'Motivation', 'Work Ethic', and 'Self-Esteem' of the selected groups.
> 2. Implement the state logic in the React component to handle multi-selection.
> 3. Ensure the design matches the existing Tailwind 'Slate/Indigo' aesthetic."

### B. Accessibility & Responsive "Print" Mode
> "The 'Print Report' feature needs to be more professional. Modify the Tailwind classes in `page.tsx` to ensure that when printing:
> 1. Background colors are removed but borders are preserved.
> 2. Each group card has a `break-inside-avoid` property.
> 3. A professional header with the date and 'Class Statistics' (Total Students, Mean Motivation) is added only to the printed version."

---

## 📝 3. Pedagogical & Strategic Communications
*Focus: Reports, stakeholder buy-in, and student engagement.*

### A. The "Psychological Safety" Syllabus Statement
> "Draft a 300-word section for a university course syllabus titled 'How Your Teams Were Formed'. Explain the GroupGen methodology to students. 
> - Focus on the concept of 'Cognitive Diversity' (combining different learning styles).
> - Reassure students that the algorithm actively prevents demographic isolation to foster a safe environment.
> - Use an encouraging, transparent tone that builds trust in the 'AI-assisted' process."

### B. Executive Summary for Administration
> "Generate a summary report for a Department Head based on the `final_groups_report.txt`. 
> 1. Quantify the 'Diversity Lift' (how much demographic isolation was reduced compared to random grouping).
> 2. Summarize the 'Balance Metrics' (Avg Motivation variance across groups).
> 3. Highlight any 'At-Risk' groups (e.g., groups with low overall Motivation) that might require additional TA support."

---

## 🧪 4. QA & Edge-Case Simulation
*Focus: Stress testing and data validation.*

### A. Synthetic Data: "The Fragmented Cohort"
> "Generate a 100-row CSV representing a 'Fragmented Cohort'. 
> - 60% of students have high 'Work_Ethic' but low 'Motivation'.
> - There are 5 different 'Diversity' categories, with 3 of them containing only a single student.
> - This dataset will be used to stress-test the `enforce_group_size` and `fix_diversity_isolation` functions. Ensure the data follows the exact GroupGen schema."

---

## 💡 5. Prompting Best Practices for GroupGen
1. **Always Attach Context:** If asking for a bug fix, paste the specific function from `clustering.py`.
2. **Chain of Thought:** Start prompts with "Let's think step-by-step about the mathematical implications of..."
3. **Structured Output:** Ask for specific formats: "Provide a JSON schema for the API response," or "Give me a Tailwind-only solution."
4. **Constraint-Based Prompting:** Explicitly state what to avoid (e.g., "Do not use external libraries like NumPy unless essential for performance").
