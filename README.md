# An AI-Integrated Multi-Stage System for Secure Clinical Note Summarization

> **MS Computer Science Thesis — California State University, Sacramento**  
> Author: Vishal Reddy Vadde | Advisor: Dr. Syed Badruddoja | Second Reader: Dr. Bang Tran

---

## Overview

This project implements a four-stage multi-agent pipeline that takes raw MIMIC-IV clinical discharge notes and produces **secure, plain-language patient summaries** while defending against prompt injection attacks and guaranteeing HIPAA-compliant PHI removal.

```
Raw Clinical Note
       │
       ▼
┌─────────────┐     Attack detected → BLOCK
│  Stage 1    │──────────────────────────────►  Rejected
│  Defender   │
└──────┬──────┘
       │ Safe input
       ▼
┌─────────────┐
│  Stage 2    │  FAISS retrieval over 929,642 chunks
│  RAG Agent  │  from 100K MIMIC-IV discharge notes
└──────┬──────┘
       │ Note + context
       ▼
┌─────────────┐
│  Stage 3    │  Fine-tuned LLaMA-3-8B-Instruct
│   Solver    │  QLoRA · BERTScore F1: 0.792
└──────┬──────┘
       │ Draft summary
       ▼
┌─────────────┐
│  Stage 4    │  Microsoft Presidio · 18 HIPAA types
│  Sanitizer  │  100% PHI detection
└──────┬──────┘
       │
       ▼
 Patient Summary
```

---

## Key Results

| Component | Metric | Result |
|-----------|--------|--------|
| Defender | Detection Rate | 92.2% |
| Defender | F1 Score | 95.9% |
| Defender | False Positive Rate | 0.0% |
| Defender | P99 Latency | 1.15 ms |
| RAG | Mean Top-1 Similarity | 0.666 |
| RAG | Avg Query Latency | 19.6 ms |
| Solver | BERTScore F1 | 0.792 |
| Solver | Training Loss | 0.5843 |
| Sanitizer | PHI Detection Rate | 100% |

---

## Repository Structure

```
clinical_mas/
├── agents/
│   ├── defender_agent.py       # Stage 1 — Prompt injection defense (104 patterns)
│   ├── rag_agent.py            # Stage 2 — FAISS retrieval pipeline
│   ├── solver_agent.py         # Stage 3 — LLaMA-3-8B summarization
│   └── sanitizer_agent.py      # Stage 4 — HIPAA PHI removal
│
├── notebooks/
│   ├── 01_defender_agent_final.ipynb       # Defender training & evaluation
│   ├── 02a_rag_knowledge_base.ipynb        # FAISS index construction
│   ├── 02b_generate_attack_dataset.ipynb   # Synthetic attack dataset generation
│   ├── 03_solver_finetune.ipynb            # LLaMA-3-8B QLoRA fine-tuning
│   ├── 04_sanitizer_agent.ipynb            # Presidio PHI sanitization
│   ├── 05_orchestrator.ipynb               # End-to-end pipeline orchestration
│   ├── 07_performance_evaluation.ipynb     # Latency & performance benchmarks
│   └── 09_model_comparison.ipynb           # Baseline model comparison
│
├── requirements.txt
└── README.md
```

> **Note:** MIMIC-IV data, fine-tuned model weights, and the attack dataset CSV are excluded from this repository due to the PhysioNet Data Use Agreement. See [Data Access](#data-access) below.

---

## Requirements

- Python 3.10+
- Google Colab Pro (recommended — A100 GPU required for fine-tuning)
- MIMIC-IV access via [PhysioNet](https://physionet.org/content/mimiciv/) (credentialed)
- Groq API key (for LLM judge evaluation)

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/VishalVadde/clinical_mas.git
cd clinical_mas
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install the core packages manually:

```bash
pip install torch transformers peft bitsandbytes \
            sentence-transformers faiss-cpu \
            presidio-analyzer presidio-anonymizer \
            spacy datasets tqdm rouge-score \
            bert-score groq
python -m spacy download en_core_web_lg
```

### 3. Mount Google Drive (if using Colab)

All notebooks are designed to run on Google Colab with files stored in Google Drive. At the top of each notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
BASE_PATH = '/content/drive/MyDrive/clinical_mas/'
```

### 4. Set Up API Keys

Create a `.env` file or set environment variables in your Colab session:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

In Colab:
```python
import os
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
```

---

## Data Access

This project uses the **MIMIC-IV** dataset (v2.2), which requires credentialed access.

1. Complete the required CITI training at [https://physionet.org](https://physionet.org)
2. Apply for access to MIMIC-IV at [https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/)
3. Once approved, download `discharge.csv` from the `note` module
4. Place the file at:

```
/content/drive/MyDrive/clinical_mas/data/discharge.csv
```

---

## Running the Pipeline

Run the notebooks **in order**. Each notebook saves its outputs to Google Drive for the next stage to consume.

### Stage 1 — Defender Agent
```
notebooks/01_defender_agent_final.ipynb
```
Builds the regex pattern library, evaluates on the attack dataset, and exports the defender module. Expected runtime: ~5 minutes on CPU.

### Stage 2a — Build RAG Knowledge Base
```
notebooks/02a_rag_knowledge_base.ipynb
```
Chunks 100,000 MIMIC-IV discharge notes into 200-word windows, embeds with `all-MiniLM-L6-v2`, and builds the FAISS IVFFlat index. Expected runtime: ~2–3 hours on Colab (index saved to Drive).

### Stage 2b — Generate Attack Dataset
```
notebooks/02b_generate_attack_dataset.ipynb
```
Generates 8,500 labeled prompt injection samples (5,000 safe + 3,500 attacks) embedded in real clinical narratives. Output is the attack dataset used to evaluate the Defender.

### Stage 3 — Fine-Tune Solver
```
notebooks/03_solver_finetune.ipynb
```
Fine-tunes `meta-llama/Meta-Llama-3-8B-Instruct` with QLoRA (r=16, α=32) on 200,000 MIMIC-IV Brief Hospital Course pairs. **Requires A100 GPU.** Expected runtime: ~4.8 hours.

> Access to LLaMA-3 weights requires accepting the Meta license at [https://huggingface.co/meta-llama](https://huggingface.co/meta-llama)

### Stage 4 — Sanitizer Agent
```
notebooks/04_sanitizer_agent.ipynb
```
Configures Microsoft Presidio with spaCy for 18 HIPAA Safe Harbor entity types and evaluates PHI removal on generated summaries.

### Stage 5 — End-to-End Orchestration
```
notebooks/05_orchestrator.ipynb
```
Runs the complete four-stage pipeline on a sample of held-out discharge notes and saves generated summaries to `results/generated_notes/`.

### Evaluation
```
notebooks/07_performance_evaluation.ipynb
notebooks/09_model_comparison.ipynb
```
Evaluates latency, BERTScore F1, ROUGE-L, and abstraction metrics. Compares against Qwen2.5-7B, Llama-3.3-70B, and Mistral-Small baselines.

---

## Using the Agent Modules Directly

The `agents/` folder contains standalone Python modules that can be imported independently:

```python
from agents.defender_agent import DefenderAgent
from agents.rag_agent import RAGAgent
from agents.solver_agent import SolverAgent
from agents.sanitizer_agent import SanitizerAgent

# Initialize
defender  = DefenderAgent()
rag       = RAGAgent(index_path="path/to/faiss.index")
solver    = SolverAgent(model_path="path/to/lora_weights")
sanitizer = SanitizerAgent()

# Run pipeline
note = "Patient was admitted for..."

result = defender.screen(note)
if result["safe"]:
    context  = rag.retrieve(note, k=5)
    summary  = solver.summarize(note, context)
    clean    = sanitizer.sanitize(summary)
    print(clean)
```

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{vadde2026clinical,
  title     = {An AI-Integrated Multi-Stage System for Secure Clinical Note Summarization},
  author    = {Vadde, Vishal Reddy},
  year      = {2026},
  school    = {California State University, Sacramento},
  type      = {MS Thesis}
}
```

### Key References

- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022.* https://arxiv.org/abs/2106.09685  
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020.* https://arxiv.org/abs/2005.11401  
- Johnson et al. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data.* https://arxiv.org/abs/1702.08734  
- Johnson, A.E.W. et al. (2023). MIMIC-IV. *PhysioNet.* https://doi.org/10.13026/6mm1-ek67  

---

## License

This repository is released for academic and research purposes only. MIMIC-IV data and derived artifacts are subject to the [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/view-license/2.2/).

---

*MS Computer Science · California State University, Sacramento · Spring 2026*
