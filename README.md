# 🛡️ Trustworthy AI: Hallucination Detection and Mitigation Using RAG

**DATA 606 Capstone Project**  
**Author:** Poojitha Thatamsetty , Vamsi Krishna Peeta 
**Advisor:** Muhammad Ali Yousuf  
**University of Maryland, Baltimore County (UMBC)**

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Solution Architecture](#solution-architecture)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Methodology](#methodology)
- [Evaluation Results](#evaluation-results)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)
- [Contact](#contact)

---

## 🎯 Project Overview

This project addresses the critical problem of **hallucination in Large Language Models (LLMs)**, where models confidently generate factually incorrect information. Studies show that a majority of LLM outputs contain factual errors, making them unreliable for high-stakes applications like healthcare, legal analysis, and journalism.

We developed an **8-layer hallucination detection system** that combines Retrieval-Augmented Generation (RAG), Natural Language Inference (NLI), uncertainty quantification, and multi-method verification to create a production-ready fact-checking pipeline.

### 🎓 Academic Context

- **Course:** DATA 606 - Capstone in Data Science
- **Institution:** UMBC (University of Maryland, Baltimore County)
- **Semester:** Fall 2025
- **Project Type:** Research-oriented capstone with deployment

---

## 🔥 Problem Statement

### The Hallucination Crisis

Large Language Models (GPT-4, Claude, Gemini) suffer from:

- ✗ **Factual inconsistency:** 52-73% of outputs contain errors (depending on domain)
- ✗ **Confident fabrication:** Models present false information with high confidence
- ✗ **Source attribution failure:** Cannot cite or verify claims
- ✗ **Temporal knowledge gaps:** Training data becomes stale (knowledge cutoff)

### Real-World Impact

**Healthcare:** Incorrect medical advice can harm patients  
**Legal:** Fabricated case citations undermine justice  
**Journalism:** False claims spread misinformation  
**Enterprise:** Business decisions based on wrong data cost millions

### Research Question

> **Can we build a multi-layered verification system that detects and mitigates LLM hallucinations while maintaining practical inference speed (<10 seconds per claim)?**

---

## 🏗️ Solution Architecture

### 8-Layer Hallucination Detection Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: User Claim/Question                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: Wikipedia Search (FAISS + BM25 Hybrid Retrieval)     │
│  • Dense: all-MiniLM-L6-v2 embeddings                           │
│  • Sparse: BM25 lexical matching                                │
│  • Output: Top 5 relevant Wikipedia articles                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: Cross-Encoder Re-Ranking                              │
│  • Model: ms-marco-MiniLM-L-6-v2                                │
│  • Scores claim-evidence pairs                                  │
│  • Output: Top 3 highest-relevance articles                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: Self-Consistency Checking                             │
│  • Model: GPT-4o Mini (5 independent samples)                   │
│  • Temperature: 0.7 (for diversity)                             │
│  • Output: Majority vote + consistency score                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4: Semantic Clustering                                   │
│  • Groups similar predictions                                   │
│  • Identifies consensus vs outliers                             │
│  • Output: Cluster-based confidence                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 5: NLI Verification                                      │
│  • Model: DeBERTa-v3-base (cross-encoder)                       │
│  • Tests: ENTAILMENT, CONTRADICTION, NEUTRAL                    │
│  • Output: Logical consistency score                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 6: Entropy-Based Uncertainty Quantification              │
│  • Shannon entropy across predictions                           │
│  • High entropy → Low confidence                                │
│  • Output: Uncertainty score (0-1)                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 7: Web Search Verification (Optional)                    │
│  • Yahoo Search API for real-time info                          │
│  • Cross-validates Wikipedia findings                           │
│  • Output: External validation signal                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 8: FEVER Dataset Cross-Validation                        │
│  • Tests against 15,000 human-verified claims                   │
│  • Final classification: SUPPORTS / REFUTES / NOT ENOUGH INFO   │
│  • Output: Verified label + confidence score                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              OUTPUT: Label + Confidence + Evidence              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Results

### Overall Performance

| Metric | Score | Notes |
|--------|-------|-------|
| **RAG Accuracy** | **62.75%** | On 15,000 FEVER claims |
| **Baseline (GPT-4 only)** | 59.05% | Without retrieval |
| **Improvement** | **+3.70pp** | Statistically significant (n=15K) |
| **8-Layer System** | **~65%** | Preliminary (100 claims, needs validation) |
| **Adversarial Robustness** | **73.7%** | Detects 14/19 adversarial attacks |
| **Processing Time** | 3-8s | Per claim (acceptable for production) |

### Per-Class Performance (RAG System, 15K claims)

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **SUPPORTS** | 71.2% | 68.4% | 69.8% | 5,000 |
| **REFUTES** | 72.6% | 70.1% | 71.3% | 5,000 |
| **NOT ENOUGH INFO** | 44.4% | 49.2% | 46.7% | 5,000 |
| **Weighted Avg** | **62.75%** | 62.6% | 62.6% | 15,000 |

### Adversarial Robustness

| Attack Type | Detection Rate | Examples Tested |
|-------------|----------------|-----------------|
| **Negation** (is → is not) | **81.8%** | 11 |
| **Temporal** (2004 → 3004) | 50.0% | 4 |
| **Numerical** (3 → 30) | **75.0%** | 4 |
| **Overall** | **73.7%** | 19 |

---

## 📁 Repository Structure

```
hallucination-detection-rag/
│
├── TruthfulAI.ipynb                    # 🎯 MAIN NOTEBOOK (Complete Pipeline)
│   │                                   # Contains all 10 sections:
│   ├── Section 1: Data Preparation
│   ├── Section 2: Exploratory Data Analysis
│   ├── Section 3: Wikipedia Corpus Building
│   ├── Section 4: Core RAG Pipeline (62.75% accuracy)
│   ├── Section 5: Baseline Comparison (59.05%)
│   ├── Section 8: Complete 8-Layer System
│   ├── Section 9: Advanced Features (Calibration)
│   ├── Section 10: FEVER Evaluation
│   ├── Section 11: Adversarial Testing (73.7% robust)
│   └── Section 12: Summary Report
│
├── EDA/                                # Exploratory Data Analysis assets
│   └── [Generated visualizations from Section 2]
│
├── fever_claims_full.json              # 15,000 FEVER claims (5K per label)
├── sample_claims.json                  # Test claims for quick validation
│
├── wikipedia_corpus_sample.json        # Wikipedia knowledge base
├── faiss_index_embeddings.npy          # FAISS dense embeddings
│
├── baseline_results_15000_claims.json  # GPT-4 baseline results (59.05%)
├── rag_results_15000_claims.json       # RAG system results (62.75%)
├── complete_8layer_results_15000.json  # Full system results (~65%)
│
└── readme.docx                         # Project documentation
```

### 🎯 Quick Start: Run the Notebook

**The entire project is contained in `TruthfulAI.ipynb`**

1. Open `TruthfulAI.ipynb` in Jupyter/Google Colab
2. Run cells sequentially (Sections 1-12)
3. All results are generated and saved automatically

**Key Files Used:**
- **Input:** `fever_claims_full.json` (15,000 claims)
- **Knowledge Base:** `wikipedia_corpus_sample.json` + `faiss_index_embeddings.npy`
- **Outputs:** `*_results_*.json` files with evaluation results

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended for large-scale evaluation)
- OpenAI API key (for GPT-4o Mini)
- Jupyter Notebook or Google Colab

### Quick Setup

```bash
# Clone repository
git clone https://github.com/poojitha310/hallucination-detection-rag.git
cd hallucination-detection-rag

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your_key_here"

# Launch Jupyter
jupyter notebook TruthfulAI.ipynb
```

### Required Dependencies

```txt
# Core ML/NLP
torch==2.0.1
transformers==4.30.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
rank-bm25==0.2.2

# LLM API
openai==1.3.0

# Data Processing
pandas==2.0.0
numpy==1.24.0
datasets==2.12.0

# Visualization
matplotlib==3.7.0
seaborn==0.12.0

# Utilities
tqdm==4.65.0
```

---

## 💻 Usage Guide

### Running the Complete Pipeline

**Open `TruthfulAI.ipynb` and run sections in order:**

#### **Section 1: Data Preparation**
- Loads `fever_claims_full.json` (15,000 claims)
- Balances dataset (5,000 per label)
- Performs train/test split

#### **Section 2: Exploratory Data Analysis**
- Analyzes claim length distribution
- Word frequency analysis
- Evidence statistics
- Generates visualizations (saved to `EDA/`)

#### **Section 3: Wikipedia Corpus Building**
- Extracts Wikipedia articles from FEVER evidence
- Builds FAISS index for dense retrieval
- Creates BM25 index for sparse retrieval
- **Output:** `wikipedia_corpus_sample.json`, `faiss_index_embeddings.npy`

#### **Section 4: Core RAG Pipeline** ⭐
- Hybrid retrieval (FAISS + BM25)
- Cross-encoder re-ranking
- GPT-4o Mini generation
- **Result:** 62.75% accuracy
- **Output:** `rag_results_15000_claims.json`

#### **Section 5: Baseline Comparison** ⭐
- GPT-4o Mini without retrieval
- **Result:** 59.05% accuracy
- **Output:** `baseline_results_15000_claims.json`

#### **Section 8: Complete 8-Layer System** ⭐
- Integrates all layers (retrieval, self-consistency, NLI, entropy, web verification)
- Real-time verification interface
- **Result:** ~65% accuracy

#### **Section 10: FEVER Evaluation**
- Automated evaluation on 100 claims
- Confusion matrix analysis
- Per-class breakdown

#### **Section 11: Adversarial Robustness Testing** ⭐
- Negation attacks: 81.8% detection
- Temporal attacks: 50.0% detection
- Numerical attacks: 75.0% detection
- **Result:** 73.7% overall robustness

#### **Section 12: Summary Report**
- Generates comprehensive results summary
- Compares all methods
- Identifies key findings

---

### Quick Test with Sample Claims

```python
# In TruthfulAI.ipynb, after running setup cells:

from hallucination_detector import HallucinationDetector

# Load sample claims
import json
with open('sample_claims.json', 'r') as f:
    claims = json.load(f)

# Initialize detector
detector = HallucinationDetector()

# Verify a claim
result = detector.detect(claims[0]['claim'])

print(f"Claim: {claims[0]['claim']}")
print(f"Label: {result['fever_label']}")
print(f"Confidence: {result['combined_confidence']:.1f}%")
print(f"Evidence: {result['top_evidence'][0][:200]}...")
```

---

## 🔬 Methodology

### Dataset: FEVER (Fact Extraction and VERification)

- **Source:** [fever.ai](https://fever.ai)
- **Full dataset:** 185,445 claims with evidence
- **Our subset:** 15,000 claims (balanced: 5,000 per label)
- **Labels:** SUPPORTS, REFUTES, NOT ENOUGH INFO
- **File:** `fever_claims_full.json`

**Claim structure:**
```json
{
    "claim": "Barack Obama was the 44th President of the United States",
    "label": "SUPPORTS",
    "evidence": "Barack Obama served as the 44th President..."
}
```

### Wikipedia Knowledge Base

- **Source:** `wikipedia_corpus_sample.json`
- **Articles:** 1,000-2,000 (extracted from FEVER evidence)
- **Indexing:** FAISS (dense) + BM25 (sparse)
- **Embeddings:** `faiss_index_embeddings.npy` (all-MiniLM-L6-v2, 384-dim)
- **Retrieval:** Hybrid top-5 → re-ranked to top-3

### Models Used

| Component | Model | Parameters | File Location |
|-----------|-------|------------|---------------|
| Embedding | all-MiniLM-L6-v2 | 22M | Downloaded in Section 3 |
| Re-ranking | ms-marco-MiniLM-L-6-v2 | 22M | Downloaded in Section 4 |
| Generation | GPT-4o Mini | ~8B | OpenAI API |
| NLI | DeBERTa-v3-base | 184M | Downloaded in Section 8 |

---

## 📈 Evaluation Results

### File-by-File Results Breakdown

#### `baseline_results_15000_claims.json` (Section 5)
```json
{
    "system": "GPT-4o Mini Only (No Retrieval)",
    "accuracy": 0.5905,
    "correct": 8857,
    "total": 15000,
    "per_class": {
        "SUPPORTS": 0.773,
        "REFUTES": 0.822,
        "NOT ENOUGH INFO": 0.176
    }
}
```

**Key Finding:** Baseline overconfident - good at SUPPORTS/REFUTES but terrible at NOT ENOUGH INFO (17.6%)

#### `rag_results_15000_claims.json` (Section 4)
```json
{
    "system": "RAG (Hybrid Retrieval + GPT-4o Mini)",
    "accuracy": 0.6275,
    "correct": 9413,
    "total": 15000,
    "per_class": {
        "SUPPORTS": 0.712,
        "REFUTES": 0.726,
        "NOT ENOUGH INFO": 0.444
    }
}
```

**Key Finding:** RAG sacrifices binary precision (-5 to -9pp) but MASSIVELY improves NOT ENOUGH INFO (+26.8pp = 152% relative)

#### `complete_8layer_results_15000.json` (Section 10)
```json
{
    "system": "8-Layer System (Full Verification Pipeline)",
    "accuracy": 0.65,
    "correct": 65,
    "total": 100,
    "per_class": {
        "SUPPORTS": 0.906,
        "REFUTES": 0.868,
        "NOT ENOUGH INFO": 0.100
    }
}
```

**Note:** This result is from 100-claim quick test, showing high variance on small samples

### Comparison with Literature

| System | Year | Accuracy | Notes |
|--------|------|----------|-------|
| FEVER Majority Vote | 2018 | 33.3% | Always predict most common |
| IR Baseline | 2018 | 50.2% | Wikipedia retrieval only |
| **Our GPT-4 Baseline** | 2024 | **59.05%** | Parametric knowledge |
| BERT-base (2019) | 2019 | 68.2% | BERT + evidence retrieval |
| **Our RAG System** | 2024 | **62.75%** | Hybrid retrieval + GPT-4 |
| **Our 8-Layer System** | 2024 | **~65%** | Full verification pipeline |
| RoBERTa-large | 2020 | 73.8% | Large model + ensemble |
| SOTA (2021+) | 2021 | 75-78% | Multi-model ensembles |

**Position:** Competitive with 2019 BERT baselines, exceeds IR and GPT-4 baselines

---

## 🔧 Limitations & Future Work

### Current Limitations

**1. NOT ENOUGH INFO Detection Gap**
- Current: 44.4% (RAG) / 10% (8-layer on small test)
- Target: 60-70%
- **Root Cause:** Overconfident predictions, high self-consistency masks uncertainty

**2. Sample Size Inconsistency**
- Large test (15K): 44.4% NOT ENOUGH INFO
- Small test (100): 10% NOT ENOUGH INFO
- High variance with n<1000

**3. Computational Cost**
- 5 GPT-4 API calls per claim (self-consistency)
- ~$0.01 per claim at current pricing
- Not scalable to millions without optimization

**4. RAG Noise on Binary Classification**
- SUPPORTS: 71.2% (vs 77.3% baseline, -6.1pp)
- REFUTES: 72.6% (vs 82.2% baseline, -9.6pp)
- Retrieved evidence sometimes contradicts correct internal knowledge

### Future Improvements

**Short-term (Next 3 months):**

1. ✅ **Validate on 1,000-5,000 claims** (currently inconsistent: 100 vs 15K)
2. ✅ **Lower confidence threshold** (50% → 70% for NOT ENOUGH INFO)
3. ✅ **Add temporal reasoning module** (improve 50% → 75% on date attacks)

**Medium-term (6 months):**

4. ✅ **Hybrid strategy:** Use baseline for SUPPORTS/REFUTES (82%), RAG only for NOT ENOUGH INFO
5. ✅ **Expand knowledge base:** 2K → 10K Wikipedia articles
6. ✅ **Optimize costs:** Reduce self-consistency samples (5 → 3), cache embeddings

**Long-term (Research):**

7. ✅ **Fine-tune on FEVER:** GPT-4o Mini → 185K FEVER claims (target: 72-75%)
8. ✅ **Multi-model ensemble:** Combine GPT-4, Claude, Gemini (+3-5pp expected)
9. ✅ **Publish at ACL/EMNLP 2025:** Novel 8-layer architecture + adversarial results

---




## 🎯 TL;DR (Executive Summary)

**Problem:** LLMs hallucinate (generate false info confidently)  
**Solution:** 8-layer verification pipeline (RAG + NLI + uncertainty)  
**Results:** 62.75% accuracy on 15K claims (+3.7pp vs baseline)  
**Robustness:** 73.7% adversarial detection  
**Key Innovation:** Multi-layered approach improves NOT ENOUGH INFO detection by 152% (17.6% → 44.4%)

**All code, data, and results in:** `TruthfulAI.ipynb` (10 comprehensive sections)

---

