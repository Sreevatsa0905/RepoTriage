# RepoTriage: Local-First GitHub Issue Intelligence & Resolution Copilot

> **RepoTriage** is an offline-first GitHub issue copilot that combines fast classical classification, hybrid lexical-dense retrieval, local LLM generation, and claim-level NLI guardrail verification to triage and resolve issues without external SaaS dependencies or ungrounded hallucinations.

---

## 1. System Architecture

```mermaid
flowchart TD
    User["Developer / Issue Reporter"] -->|"Issue Text or URL"| CLI["RepoTriage CLI / FastAPI"]
    
    subgraph S0 ["Stage 0: Classification (Sub-10ms)"]
        CLI --> CLF["Frozen TF-IDF + LogReg<br/>(Pooled NLBSE 2024)"]
        CLF -->|"P(class) >= 0.50"| ClfLabel["Label: bug | feature | question"]
        CLF -->|"P(class) < 0.50"| ClfUncertain["Label: uncertain (needs_review)"]
    end

    subgraph S1 ["Stage 1 & 2: Hybrid Retrieval (Issues & Docs)"]
        CLI --> BM25["BM25 Lexical Search<br/>(Code-Aware Tokenizer)"]
        CLI --> Dense["BGE-Base-en-v1.5 Dense<br/>(Cached Embeddings)"]
        BM25 -->|"Top 50"| RRF["Reciprocal Rank Fusion<br/>(k = 60)"]
        Dense -->|"Top 50"| RRF
        RRF -->|"Top 20"| Rerank["BGE-Reranker-Base<br/>(Score Blend α = 0.50)"]
        Rerank --> Evidence["Top 6 Evidence Chunks<br/>(E1..E6: Issues & Docs)"]
    end

    subgraph S3 ["Stage 4: Local LLM Synthesis"]
        Evidence --> Qwen["Qwen 3.5 4B (Ollama)<br/>RTX 3050 6GB Laptop GPU"]
        CLI --> Qwen
        Qwen --> GenOut["Structured Triage JSON<br/>- Summary & Root Cause<br/>- Resolution Steps<br/>- Harvested Claims + Citations"]
    end

    subgraph S4 ["Stage 5 & 6: Claim-Level Guardrail Verification"]
        GenOut --> NLI["DeBERTa-v3 NLI Cross-Encoder<br/>(Multi-Premise Competition)"]
        Evidence --> NLI
        NLI --> Dominance{"Dominance Rule<br/>τ_s = 0.80, τ_c = 0.80"}
        Dominance -->|"p_e >= 0.80 & p_e > p_c"| Supp["SUPPORTED (Green)"]
        Dominance -->|"p_c >= 0.80 & p_c > p_e"| Cont["CONTRADICTED (Red)"]
        Dominance -->|"Otherwise"| Insuff["INSUFFICIENT_EVIDENCE (Amber)"]
    end

    Supp --> Out["Final Verified Report & Interactive UI"]
    Cont --> Out
    Insuff --> Out
```

---

## 2. Key Empirical Results

### A. Retrieval 4-Way Ablation (Issues-Only Candidate Pool, $n=51$, 1,000 Bootstrap Resamples)

| Retrieval Method | Recall@5 [95% CI] | Precision@5 | MRR@10 [95% CI] | nDCG@10 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Code-Aware)** | 0.7451 [0.6275, 0.8627] | 0.1765 | 0.5484 [0.4340, 0.6670] | 0.5977 | 1.2 ms |
| **Dense (BGE-Base-en-v1.5)** | 0.8039 [0.6863, 0.9020] | 0.1843 | 0.5730 [0.4675, 0.6891] | 0.6309 | 12.5 ms |
| **BM25 + Dense (RRF, $k=60$)** | 0.7647 [0.6471, 0.8824] | 0.1765 | 0.6018 [0.4948, 0.7139] | 0.6429 | 14.1 ms |
| **RRF + Reranker (Blend $\alpha=0.50$)** | **0.7843** [0.6667, 0.9020] | 0.1804 | **0.6022** [0.4946, 0.7140] | **0.6481** | 74.8 ms |

> **Note on Retrieval**: Restricting the candidate pool to issues and comments accounted for the primary jump in Recall@5 (from ~0.45 to ~0.78). RRF and reranker blending provide marginal consistency gains (+0.0392 Recall@5, +0.0538 MRR@10) with 95% paired difference CIs overlapping zero.

### B. Guardrail In-Domain Calibration (B2 Fresh Held-Out Set, $n = 80$ Claims)

> [!NOTE]
> **Claim Sets Reconciliation**:
> - **Stage 6.5 Exploratory Dev Set**: Initial testing used an exploratory set of 10 issue queries (~50 claims total) from early pipeline prototypes, establishing preliminary pass rates (24.0% at $\tau = 0.30$ under single-chunk premise).
> - **Stage 6.6 Standardized Evaluation Benchmark**: Constructed a standardized, strictly held-out evaluation set of 15 issues across 4 repositories (`encode/httpx`: 4, `pallets/flask`: 4, `pydantic/pydantic`: 4, `psf/requests`: 3) with zero dev overlap. This produced exactly **80 claims** ($n = 80$, 5.33 claims/issue). 75 of 80 claims (93.75%) cited at least one evidence item, while 5 claims (6.25%) had missing or invalid citations and were immediately rejected as `INSUFFICIENT_EVIDENCE` by rule.
> - **Stage 6.6-D Synthetic Perturbation Benchmark**: A separate rule-constructed synthetic benchmark of **300 pairs** ($n = 300$, balanced: 100 SUPPORTED, 100 CONTRADICTED, 100 INSUFFICIENT_EVIDENCE).

| Threshold Setting | Grounded Pass Rate (Cited Claims) | Grounded Pass Rate (All Claims) | Easy Neg False-Support | Hard Neg False-Support | Contradiction Detection |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\tau_s = 0.50, \tau_c = 0.50$ | 42.67% (32 / 75) | 40.00% (32 / 80) | 1.25% (1 / 80) | 0.00% (0 / 80) | 15.00% (12 / 80) |
| $\tau_s = 0.70, \tau_c = 0.70$ | 40.00% (30 / 75) | 37.50% (30 / 80) | 1.25% (1 / 80) | 0.00% (0 / 80) | 15.00% (12 / 80) |
| **$\tau_s = 0.80, \tau_c = 0.80$ (Production)** | **40.00% (30 / 75)** | **37.50% (30 / 80)** | **1.25% (1 / 80)** | **0.00% (0 / 80)** | **15.00% (12 / 80)** |
| $\tau_s = 0.90, \tau_c = 0.90$ | 34.67% (26 / 75) | 32.50% (26 / 80) | 0.00% (0 / 80) | 0.00% (0 / 80) | 12.00% (10 / 80) |

*Note: In addition to the full 80-pair hard negative evaluation (0 / 80, 0.00%), a detailed manual audit spot-check was conducted on 10 hard negative pairs (0 / 10 false support, 0.00%).*

### C. Automated Perturbation Test (Synthetic Rule-Constructed Ground Truth, $n=300$)
- **Multi-Premise Production Accuracy ($\tau_s = 0.80, \tau_c = 0.80$)**: **88.67%** (266 / 300) | **Macro-F1: 0.8862**
- **SUPPORTED Recall**: **95.00%** (95 of 100 verbatim code assertions approved) | **Precision: 94.06%**
- **CONTRADICTED Recall**: **92.00%** (92 of 100 negated assertions detected) | **Precision: 81.42%**
- **INSUFFICIENT_EVIDENCE Recall**: **79.00%** (79 of 100 unrelated cross-repo chunks rejected) | **Precision: 91.86%**
- **False-Support Rate on Contradicted**: **6.00%** (6 / 100)
- **False-Support Rate on Unrelated (NEI)**: **0.00%** (0 / 100)
- **Single-Chunk Premise Ablation (Without Windowing)**: Accuracy = 79.33% (238 / 300), SUPPORTED Recall = 67.00% (67 / 100) due to premise attention dilution.
- **Human Check Status**: **not performed**

### D. Classifier Live Ingest Generalization ($n=159$ Ingested Issues Across 3 Repos)
- **Coverage at $\tau = 0.50$**: **73.58%** (117 / 159 covered; 42 routed to "uncertain")
- **Performance on Covered Issues**: **82.05% Accuracy**, **Macro-F1: 0.6979** (Bug F1: 0.7222, Feature F1: 0.9048, Question F1: 0.4667)
- **Cold-Start LOGO CV**: 0.5941 Macro-F1 | **Pooled NLBSE Test**: 0.7605 Macro-F1 (76.07% Accuracy)

---

## 3. UI Screenshots

### Live Issue Copilot: Input & Staging (Tab 1)
![Issue Copilot Initial View](docs/screenshots/ui_copilot.png)

### Completed Multi-Stage Analysis Result & Verified Claims
![Completed Analysis Result](docs/screenshots/ui_result.png)

### Evaluation & Benchmarks Dashboard (Tab 2)
![Evaluation Dashboard](docs/screenshots/ui_evaluation.png)

---

## 4. Quickstart

### Prerequisites
- Python 3.10+
- Node.js 18+ & npm
- [Ollama](https://ollama.ai) installed and running with Qwen 4B:
  ```powershell
  ollama serve
  ollama pull qwen3.5:4b
  ```

### Installation
```powershell
# 1. Clone repository and setup Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Setup environment variables
cp .env.example .env
# Edit .env with your GITHUB_TOKEN

# 3. Setup Frontend
cd frontend
npm install
cd ..
```

### Running RepoTriage

#### Option A: CLI Interface
```powershell
# Single-line issue classification
python -m src.cli classify "Using httpx with proxy fails with SSL: UNEXPECTED_MESSAGE"

# Hybrid search over indexed chunks
python -m src.cli search "field_validator Optional NoneType" --repo pydantic/pydantic

# Full triage pipeline with LLM synthesis and NLI verification
python -m src.cli triage "field_validator Optional NoneType error" --repo pydantic/pydantic
```

#### Option B: Full-Stack Web Application
```powershell
# Terminal 1: Launch FastAPI Backend (Port 8000)
.\.venv\Scripts\uvicorn.exe src.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Launch Vite React Frontend (Port 5173)
cd frontend
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 5. End-to-End Reproducibility Guide

To reproduce the entire experimental pipeline from corpus acquisition to final benchmarks:

```powershell
# 1. Acquire corpus and clone documentation (Stage 1)
python -m src.cli ingest --all

# 2. Build retrieval ground-truth pairs (Stage 3)
python -m eval.build_retrieval_set

# 3. Execute 4-way retrieval ablation & bootstrap statistics (Stage 3 & 6.6)
python -m eval.run_retrieval_eval
python -m eval.compute_retrieval_statistics

# 4. Run classifier live ingest check (Stage 6.6)
python -m eval.run_classifier_live_check

# 5. Run automated perturbation test (Stage 6.6)
python -m eval.run_perturbation_test

# 6. Run guardrail evaluation (B1 ANLI + B2 negative controls) (Stage 6.6)
python -m eval.run_guardrail_eval

# 7. Run full offline test suite
python -m unittest discover tests
```

---

## 6. Technical Resume Entry Draft

> **Local-First GitHub Issue Intelligence & Resolution Copilot (RepoTriage)**
> * Built an offline-first issue resolution copilot combining a selective TF-IDF + Logistic Regression classifier (73.58% coverage at 0.50 confidence, achieving 82.05% accuracy on covered issues), hybrid BM25 + BGE dense retrieval (Dense Recall@5: 0.8039 [95% CI: 0.6863, 0.9020]; RRF MRR@10: 0.6018 [95% CI: 0.4948, 0.7139] on $n=51$ duplicate query pairs), local Qwen 4B synthesis, and DeBERTa-v3 natural language inference (NLI) claim verification.
> * Discovered that an off-the-shelf cross-encoder reranker (`bge-reranker-base`) degraded pure lexical-dense Recall@5 from 0.7647 to 0.7451 until rescued by convex reciprocal-rank score blending ($\alpha = 0.50$, recovering 0.7843 Recall@5 and 0.6022 MRR@10).
> * Implemented multi-premise sentence-window NLI verification enforcing strict dominance thresholds ($\tau_s = 0.80, \tau_c = 0.80$), eliminating premise-attention dilution and achieving 0.00% (0 / 80) false-support on hard negative controls, 1.25% (1 / 80) on easy negative controls, and 95.00% (95 / 100) supported recall on synthetic rule-perturbed developer claims.
> * Engineered deterministic GPU memory orchestration for consumer hardware (NVIDIA RTX 3050 6GB), decoupling CPU-resident dense index querying (<15ms) from sequential VRAM-isolated LLM generation and NLI verification.
> * Built an asynchronous FastAPI REST service with single-flight concurrency locking (HTTP 429) alongside a Vite React + TypeScript dark-theme copilot displaying unmerged dual classification and interactive claim verification traces.

---

## 7. Project Documentation & History

Detailed technical specifications, milestone progress, and architecture decision records are maintained in:
- [System State & Engineering Journal](docs/CURRENT_STATE.md)
- [Comprehensive Empirical Evaluation Report](EVALUATION.md)
- [Datasets & Corpus Specification](DATASETS.md)
- [Model Cards & Memory Specifications](MODEL_CARD.md)
