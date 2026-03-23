# Multi-Agent System for Secure Clinical Summarization

A multi-agent pipeline for secure, HIPAA-compliant clinical summarization of MIMIC-IV discharge notes.

## Architecture

```
Clinical Discharge Note
        ↓
[Defender Agent]   — Prompt injection detection (97 patterns, 7 attack categories)
        ↓
[RAG Agent]        — FAISS vector retrieval from MIMIC-IV knowledge base
        ↓
[Solver Agent]     — LLaMA-3-8B + LoRA fine-tuned patient summary generation
        ↓
[Sanitizer Agent]  — Microsoft Presidio PHI detection & HIPAA compliance
        ↓
    Patient Summary
```

## Notebooks

| Notebook | Description |
|---|---|
| `01_defender_agent_final.ipynb` | Rule-based prompt injection defender |
| `02a_rag_knowledge_base.ipynb` | FAISS index construction from MIMIC-IV |
| `02b_generate_attack_dataset.ipynb` | Synthetic attack dataset generation (8,500 samples) |
| `03_solver_finetune.ipynb` | LLaMA-3-8B LoRA fine-tuning |
| `04_sanitizer_agent.ipynb` | PHI detection and anonymization |
| `05_orchestrator.ipynb` | Full pipeline integration |
| `06_evaluation.ipynb` | System evaluation (ROUGE-L, F1, latency) |
| `07_performance_evaluation.ipynb` | Detailed performance analysis |
| `demo.ipynb` | End-to-end demo — paste any discharge note |

## Data Requirements

This project uses the **MIMIC-IV** clinical dataset, which requires credentialed access.  
To reproduce results, you must apply for access at [PhysioNet](https://physionet.org/content/mimiciv/2.2/).  
The data itself is **not included** in this repository.

## Tech Stack

- **Model:** Meta LLaMA-3-8B-Instruct (4-bit quantized, LoRA rank=16)
- **Retrieval:** FAISS + Sentence-Transformers (all-MiniLM-L6-v2)
- **PHI Detection:** Microsoft Presidio + spaCy
- **Infrastructure:** Google Colab Pro, Google Drive

## Paper

See `main.pdf` for the full project report.
