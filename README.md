# LegalIR-PCSR 🏛️

## Automated Legal Information Retrieval with BM25, SBERT, Ensemble and RAG
## Evaluated on IL-PCSR Benchmark (EMNLP 2025)

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Dataset](https://img.shields.io/badge/Dataset-IL--PCSR-orange)]()
[![Target](https://img.shields.io/badge/Target-FIRE2026-red)]()
[![University](https://img.shields.io/badge/University-Air%20University-blue)]()

---

## 📋 Overview

**LegalIR-PCSR** is an automated Information Retrieval (IR) system for legal statutes and prior case retrieval. It replicates and extends the IL-PCSR benchmark (Paul et al., EMNLP 2025) by implementing a complete multi-stage retrieval pipeline:

- ✅ **BM25 lexical baseline** (2, 3, 4, 5 n-grams) — exactly as IL-PCSR paper
- ✅ **SBERT semantic retrieval** — all-MiniLM-L6-v2 model
- ✅ **BM25 + SBERT ensemble** — with grid search for optimal alpha
- ✅ **RAG pipeline** — citizen-friendly plain English explanations *(original contribution — not in IL-PCSR paper)*

The system allows citizens to describe their legal situation in plain natural language, retrieves the most relevant statutes and court precedents, and generates a plain English explanation using a Large Language Model.

---



## 📊 Results

### Our Results vs IL-PCSR Paper (Paul et al., EMNLP 2025)

| Method | Our LSR F1 | Paper LSR F1 | Our PCR F1 | Paper PCR F1 |
|---|---|---|---|---|
| BM25 2-gram | 11.13% | 13.15% | 20.65% | 24.89% |
| BM25 3-gram | **15.42%** | **17.80%** | 27.15% | 32.21% |
| BM25 4-gram | 16.51% | 17.06% | 30.15% | 30.54% |
| BM25 5-gram | 15.87% | 16.98% | **31.07%** | **33.29%** |
| SBERT | 5.37% | 7.15% | 3.46% | 9.94% |
| Ensemble (Best) | 15.42% | 36.17% | 31.28% | 36.35% |
| **RAG Pipeline** | **✅ Implemented** | **❌ Not in paper** | **✅ Implemented** | **❌ Not in paper** |

### Key Findings
- ✅ BM25 3-gram is best for LSR — confirms paper finding
- ✅ BM25 5-gram is best for PCR — confirms paper finding
- ✅ Lexical methods outperform semantic for PCR — confirms paper finding
- ✅ Ensemble improves PCR over pure BM25
- 🆕 RAG pipeline generates citizen-friendly legal explanations — original contribution

---

## 🗄️ Dataset

We use the **IL-PCSR dataset** (Paul et al., EMNLP 2025):

| Component | Size | Description |
|---|---|---|
| Statute pool | 936 statutes | Articles/Sections from 92 Central Acts |
| Precedent pool | 3,183 cases | Supreme Court + High Court judgments |
| Test queries | 627 cases | Case judgment documents |
| Train queries | 5,017 cases | For model training |
| Dev queries | 627 cases | For validation |

**Dataset link:** 👉 [huggingface.co/datasets/Exploration-Lab/IL-PCSR](https://huggingface.co/datasets/Exploration-Lab/IL-PCSR)

> Note: Dataset requires HuggingFace account and access approval (automatic).

---

## 🚀 How to Run

### Prerequisites
- Google Colab (free tier works)
- HuggingFace account + token
- Groq account + API key (free at console.groq.com)

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/LegalIR-PCSR
cd LegalIR-PCSR
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Add API Keys to Colab Secrets
```
HF_TOKEN      → HuggingFace token
               Get at: huggingface.co/settings/tokens

GROQ_API_KEY  → Groq API key
               Get at: console.groq.com
```

### 4. Run Notebooks in Order

| Notebook | Purpose | Runtime | Time |
|---|---|---|---|
| LegalIR_01_Data.ipynb | Load IL-PCSR dataset | CPU | ~5 min |
| LegalIR_02_BM25.ipynb | BM25 baseline | CPU | ~3 hrs |
| LegalIR_03_SBERT.ipynb | SBERT embeddings | **GPU** | ~15 min |
| LegalIR_04_Ensemble.ipynb | Ensemble + grid search | **GPU** | ~1 hr |
| LegalIR_05_RAG.ipynb | RAG pipeline | CPU | ~20 min |
| LegalIR_06_Results.ipynb | Results table | CPU | ~2 min |

> ⚠️ Switch to GPU runtime (T4) for Notebooks 3 and 4

### 5. Saved Files Between Notebooks

```
LegalIR_01 → processed_data.pkl
LegalIR_02 → bm25_results.pkl
LegalIR_03 → statute_embeddings.pt
             precedent_embeddings.pt
             query_embeddings.pt
LegalIR_04 → all_results.pkl
LegalIR_05 → rag_results.json
             rag_results.pkl
```

> Download each pkl/pt file to laptop after each notebook. Upload to next notebook.

---



## 🔧 Pipeline Architecture

### Evaluation Pipeline
```
User Query
    │
    ├──→ BM25 (2,3,4,5 n-grams)
    │         └──→ Rank 936 statutes  → F1@k / MAP / MRR  [LSR]
    │         └──→ Rank 3183 prec.    → F1@k / MAP / MRR  [PCR]
    │
    ├──→ SBERT (all-MiniLM-L6-v2)
    │         └──→ Cosine similarity  → F1@k / MAP / MRR  [LSR + PCR]
    │
    └──→ Ensemble (alpha grid search 0.0 → 1.0)
              └──→ Z-Norm(BM25) + Z-Norm(SBERT)
                        └──→ Best alpha: LSR=0.0 / PCR=0.5
                                  └──→ F1@k / MAP / MRR  [LSR + PCR]
```

### RAG Pipeline (Original Contribution)
```
User Query (plain English)
    │
    ├──→ BM25 3-gram → top-3 statutes  ──┐
    │                                     │
    └──→ BM25 5-gram → top-3 precedents ─┤
                                          │
                                          ▼
                               Groq LLaMA-3.3-70b-versatile
                               (6 documents as context)
                                          │
                                          ▼
                               Plain English Explanation
                               + Practical Steps
                               + Legal Disclaimer ⚠️
```

---

## 💡 Original Contribution

The IL-PCSR paper (Paul et al., EMNLP 2025) implemented BM25, GNN-based semantic models, and LLM re-ranking for retrieval evaluation. **It did NOT implement a RAG pipeline for citizen-facing explanations.**

Our original contribution:

> We implement a **citizen-facing RAG pipeline** that:
> 1. Retrieves top-3 relevant statutes using BM25 3-gram
> 2. Retrieves top-3 relevant precedents using BM25 5-gram
> 3. Passes all 6 documents as context to Groq LLaMA-3.3-70b
> 4. Generates plain English legal explanations
> 5. Includes actionable steps and legal disclaimer
>
> This extends the IL-PCSR benchmark toward **practical legal assistance for common citizens** who cannot afford professional legal counsel.

---

## 🧪 RAG Examples

### Query 1 — Employment Issue
```
Input: "My employer has not paid my salary for 3 months.
        What law protects me?"

Output: "The laws that apply to your situation are related
         to employment and payment of salaries. Statute 3
         (ID: 938899) defines what constitutes a salary...

         Practical steps:
         1. Send a formal letter demanding payment
         2. Keep records of all communication
         3. File complaint with labour department

         ⚠️ DISCLAIMER: Not legal advice.
            Consult a qualified lawyer."
```

### Query 2 — Tenant Issue
```
Input: "My landlord cut my electricity without notice.
        What are my rights?"

Output: "The landlord's action is likely a violation of
         your rights as a tenant. Landlords are required
         to provide safe and habitable living conditions...

         Practical steps:
         1. Document the incident with dates and times
         2. Review your lease agreement
         3. Contact local housing authority

         ⚠️ DISCLAIMER: Not legal advice."
```

### Query 3 — Consumer Fraud
```
Input: "I bought a mobile phone online and received a
        fake item. Seller is refusing to refund.
        What can I do legally?"

Output: "Statute 2 (ID: 1454268) and Statute 3
         (ID: 1538044) apply to your situation...

         Practical steps:
         1. Contact the seller again formally
         2. File complaint with consumer protection agency
         3. Consider small claims court

         ⚠️ DISCLAIMER: Not legal advice."
```

---

## 🔮 Future Work

- [ ] Apply pipeline to Pakistani legal corpus (PPC + Supreme Court of Pakistan)
- [ ] Create  Pakistani Legal IR benchmark — PakLegal-IR
- [ ] Bilingual retrieval supporting English + Urdu queries
- [ ] Fine-tune SBERT on IL-PCSR training set
- [ ] Implement Para-GNN for improved semantic retrieval
- [ ] Add ensemble-based retrieval to RAG pipeline


---

## 📚 References

**Primary Paper:**
```bibtex
@inproceedings{paul2025ilpcsr,
  title     = {IL-PCSR: Legal Corpus for Prior
               Case and Statute Retrieval},
  author    = {Paul, Shounak and Ghumare,
               Dhananjay and Goyal, Pawan and
               Ghosh, Saptarshi and Modi, Ashutosh},
  booktitle = {Proceedings of EMNLP 2025},
  pages     = {14588--14611},
  year      = {2025}
}
```

**Other References:**
- Robertson, S. & Zaragoza, H. (2009). BM25 and Beyond.
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT.
- World Justice Project. (2024). Rule of Law Index.

---

## ⚠️ Disclaimer

This system is for **research and educational purposes only**.

All outputs generated by the RAG pipeline are informational and do **not** constitute legal advice. The system retrieves information from the IL-PCSR dataset which contains Indian legal documents.

**Always consult a qualified lawyer for your specific legal situation.**

---

## 📜 License

MIT License — Copyright (c) 2026 Khalid Rafique, Ismail Hamza, Muhammad Mutahir — Air University Islamabad

See LICENSE file for full details.





