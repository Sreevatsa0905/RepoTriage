# RepoTriage Model Card & Hardware Architecture

This document specifies the models powering RepoTriage, their memory footprints, execution schedules on resource-constrained consumer GPUs (NVIDIA RTX 3050 6GB Laptop GPU), intended operating regimes, and known failure modes.

---

## 1. System Model Registry

| Model Name | Role | Architecture | Parameters / Weights | Execution Device | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`RepoTriage-Pooled-LogReg`** | Issue Classification | TF-IDF (word+char) + Logistic Regression | ~5.0 MB joblib | CPU | ~50 MB RAM |
| **`BAAI/bge-base-en-v1.5`** | Dense Retrieval Embedder | BERT-base Bi-Encoder | 109M parameters (~436 MB) | CPU / CUDA | ~450 MB VRAM |
| **`BAAI/bge-reranker-base`** | Candidate Reranking | RoBERTa-base Cross-Encoder | 110M parameters (~440 MB) | CPU / CUDA | ~500 MB VRAM |
| **`qwen3.5:4b` (via Ollama)** | Structured Synthesis & Claim Extraction | Qwen2.5 / Qwen3.5 Transformer | 4.0B parameters (~2.6 GB Q4_K_M) | CUDA (RTX 3050) | ~3,100 MB VRAM |
| **`cross-encoder/nli-deberta-v3-base`**| Claim Guardrail Verification | DeBERTa-v3-base Cross-Encoder | 86M parameters (~345 MB) | CUDA (RTX 3050) | ~850 MB VRAM |

---

## 2. RTX 3050 6GB Laptop GPU Memory Management

Consumer GPUs with 6GB of VRAM cannot simultaneously hold a 4B parameter LLM, a dense embedder, a cross-encoder reranker, and a DeBERTa NLI verifier without triggering Out-Of-Memory (OOM) faults or catastrophic CUDA paging.

### Sequential Memory Lifecycle
RepoTriage implements a deterministic, multi-stage memory orchestration pipeline:

```
[Phase 1: Ingestion & Embeddings]
  └── Pre-computed offline: Corpus BGE embeddings saved as float32 .npy arrays on disk.
      During query search, single query vector is embedded on CPU in <15ms.

[Phase 2: Hybrid Retrieval & Reranker]
  └── Reranker (BAAI/bge-reranker-base) runs on top-20 RRF candidates on CPU/GPU.
      Total retrieval latency: ~75ms.

[Phase 3: LLM Generation (Ollama)]
  └── Ollama server handles Qwen 4B generation (~2.6GB VRAM footprint).
      Prompt enforces strict, anti-meta-language claim generation with evidence citations.

[Phase 4: Guardrail Verification (DeBERTa-v3)]
  └── Loaded dynamically on CUDA.
      Runs multi-premise NLI verification over harvested claims.
      Explicit CUDA cache flushing (torch.cuda.empty_cache()) after batch completion.
```

---

## 3. Detailed Model Profiles & Intended Use

### Model 1: RepoTriage Frozen Classifier
- **Intended Use**: Instant, sub-10ms triage categorizing issues into `bug`, `feature`, or `question`.
- **Strengths**: Robust across standard developer vocabulary; zero GPU requirement.
- **Failure Modes**: Domain shift when applied to specialized library syntaxes. Mitigated by setting a minimum confidence threshold ($\tau = 0.50$) to classify low-certainty predictions as `uncertain`.

### Model 2: BAAI/bge-base-en-v1.5 (Dense Embedder)
- **Intended Use**: Semantic nearest-neighbor retrieval capturing paraphrased problem descriptions.
- **Strengths**: High conceptual recall (`Recall@5 = 0.8039` on held-out issues).
- **Failure Modes**: Semantic dilution on precise error traces, hex memory addresses, and version numbers. Fused with BM25 via Reciprocal Rank Fusion (RRF, $k=60$) to ensure exact keyword fidelity.

### Model 3: BAAI/bge-reranker-base (Cross-Encoder)
- **Intended Use**: Joint query-document attention scoring of top fused candidates.
- **Strengths**: Strong contextual relevance discrimination.
- **Failure Modes**: Over-confidence on short titles; mitigated via convex linear score blending with RRF rank scores ($\alpha = 0.50$).

### Model 4: Qwen 3.5 4B (Local Generation via Ollama)
- **Intended Use**: Extraction of atomic, testable propositions and synthesis of root cause analysis without external API dependencies.
- **Strengths**: High local inference throughput (~25-30 tok/s on RTX 3050); excellent instruction following for structured JSON schemas.
- **Failure Modes**: Hallucination of version numbers or subtle parameter behavior when documentation is sparse.

### Model 5: cross-encoder/nli-deberta-v3-base (NLI Guardrail)
- **Intended Use**: Natural language inference classifying (Premise, Claim) pairs into `entailment`, `contradiction`, or `neutral`.
- **Decision Rule**: Strict dominance rule ($p_e \ge 0.80 \land p_e > p_c$ for `SUPPORTED`, $p_c \ge 0.80 \land p_c > p_e$ for `CONTRADICTED`).
- **Failure Modes**: Rigid syntactic interpretation of code blocks or truncated stack traces (accounting for 5/12 audit false alarms). Mitigated via multi-premise competition and sentence-level window extraction.
