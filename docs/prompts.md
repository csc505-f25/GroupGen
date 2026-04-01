# GroupGen: The Prompt Engineering Library

This library is a collection of high-fidelity prompts designed to orchestrate the **GroupGen** ecosystem. Use these templates to guide LLMs through the transition from tabular logic to a **Multimodal Deep Learning** architecture.

---

## 🏗️ 0. Global System Context (The "System Persona")
*Copy and paste this at the start of any new chat session to provide the AI with full architectural awareness.*

> **Persona:** You are a Senior Fullstack Engineer and Deep Learning Researcher specializing in EdTech and Multimodal AI.
>
> **Project Overview:** "GroupGen" is transitioning from a tabular clustering engine to a **Multimodal Deep Learning System**. It uses a two-branch Autoencoder to fuse structured survey data with unstructured natural language text into a unified 16-dimensional latent space.
> - **Backend:** Python, FastAPI, PyTorch, HuggingFace Transformers (DistilBERT), Scikit-learn.
> - **Architecture:** A Multimodal Autoencoder compresses tabular and text data into bottleneck embeddings, which are then clustered via K-Medoids.
> - **Key Innovation:** Semantic matching of student "Working Styles" paired with a post-clustering "Locking Mechanism" to ensure demographic fairness and prevent isolation.
> - **Frontend:** Next.js 14+, TypeScript, Tailwind CSS, Recharts.

---

## 🧠 1. Phase-Based Multimodal Implementation
*Focus: Building the Deep Learning pipeline from scratch.*

### A. Phase 1: Environment & Deep Learning Dependencies
> "We are upgrading GroupGen to support Multimodal Deep Learning. 
> 1. Update `requirements.txt` to include `torch`, `transformers`, and `tqdm`. 
> 2. Propose a directory structure for the new ML assets (e.g., `backend/models/`, `backend/scripts/`, `backend/weights/`).
> 3. Verify that `distilbert-base-uncased` is the optimal choice for local training on a standard laptop vs. a full BERT model."

### B. Phase 2: Procedural Synthetic Data Generation
> "We need 5,000 synthetic student profiles to train our Autoencoder. Create `backend/scripts/generate_multimodal_data.py` to:
> 1. Generate random tabular traits (Gender, Diversity, Motivation, etc.) following existing distributions.
> 2. Use a template-based procedural approach to generate 'Working Style' paragraphs. 
> 3. **CRITICAL:** Ensure the text content correlates with the tabular traits (e.g., a student with high 'Work Ethic' should have text suggesting leadership or diligence) so the model has a meaningful manifold to learn.
> 4. Export the result to `backend/data/synthetic_multimodal_students.csv`."

### C. Phase 3: The Two-Branch Autoencoder Architecture
> "Design the PyTorch architecture in `backend/multimodal_autoencoder.py`.
> 1. **TabularBranch:** A MLP that handles one-hot encoded and scaled survey data.
> 2. **TextBranch:** Uses a frozen `DistilBertModel` to extract the `[CLS]` token embedding.
> 3. **Fusion Layer:** Concatenates both branches and reduces them to a 16-dimensional bottleneck.
> 4. **Decoder:** Reconstructs the **Tabular targets only** (reconstructing text is too expensive). 
> 5. Explain why using the text solely for the bottleneck compression (and not reconstruction) is standard practice for multimodal clustering."

### D. Phase 4: Training Loop & Convergence Strategy
> "Implement the training pipeline in `backend/scripts/train_autoencoder.py`.
> 1. Create a `StudentDataset` class that handles tokenization via `DistilBertTokenizer`.
> 2. Write a training loop with `MSELoss` for tabular reconstruction.
> 3. Include `tqdm` progress bars and periodic validation on an 80/20 split.
> 4. Save the final model weights to `backend/weights/multimodal_autoencoder.pt`. 
> 5. Describe how to monitor the loss curve to ensure the network isn't just memorizing the tabular inputs."

### E. Phase 5: Inference & K-Medoids Integration
> "Integrate the trained model into the production pipeline via `backend/multimodal_inference.py`.
> 1. Load the frozen `.pt` weights.
> 2. Create an `extract_embeddings(df)` function that returns the 16D bottleneck matrices.
> 3. Update `backend/clustering.py` to use these embeddings for distance calculation instead of the raw tabular data.
> 4. Ensure the 'Locking Mechanism' (Heuristic Swaps) still functions correctly on the resulting clusters."

---

## 🧬 2. Classic Algorithmic Tuning (Legacy Support)
*Focus: Refining the existing K-Medoids and Heuristic logic.*

### A. Weighted Gower Distance
> "We need to adjust the clustering to weigh 'Learning_Style' more heavily. Propose a custom distance function in `backend/kmedoids.py` that applies a 2.0x weight to mismatches in categorical traits while keeping numerical scores normalized."

### B. Simulated Annealing Swap Logic
> "Replace the greedy 'Locking Mechanism' in `backend/clustering.py` with a **Simulated Annealing** heuristic. The energy function should minimize demographic isolation while preventing a significant increase in cluster variance (WCSS)."

---

## 🎨 3. Frontend & UI/UX Orchestration
*Focus: Visualizing the new Multimodal clusters.*

### A. Embedding Space Visualization (PCA)
> "I want to visualize the 16D student embeddings on the frontend. 
> 1. Create a FastAPI endpoint that runs PCA on the student embeddings to reduce them to 2D coordinates.
> 2. On the frontend, use **Recharts** (ScatterChart) to plot these students, coloring them by their assigned cluster.
> 3. Allow users to hover over a point to see the student's original 'Working Style' text."

---

## 💡 4. Prompting Best Practices for GroupGen
1. **Chain of Thought:** Always ask the AI to "think step-by-step about the latent space implications."
2. **Context Injection:** When debugging the Autoencoder, provide the shapes of your input tensors (e.g., "The tabular tensor is [Batch, 12]").
3. **Weight Management:** Explicitly remind the AI to handle `model.eval()` and `torch.no_grad()` during inference to save memory.
4. **Constraint Awareness:** Remind the AI that while the *distance* is now deep-learned, the *group size* and *diversity rules* are still hard-coded heuristics.
