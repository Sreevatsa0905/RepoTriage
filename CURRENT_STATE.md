# RepoTriage — Current Repository State

**Last Updated**: November 2026  
**Specification**: `RepoTriage_SPEC.md`  
**Current Milestone**: Stages 0, 1, 2, and 3 COMPLETE.

---

## 1. Executive Summary

- **Hardware Status**: GPU power state issue resolved (NVIDIA driver updated to 621.14). The RTX 3050 Laptop GPU operates in **P0 Maximum Performance State** with a **70W power ceiling** (Graphics: 1,492–2,100 MHz, Memory: 7,001 MHz).
- **Unit Tests**: **60 automated tests passing in 0.55s**.
- **Completed Stages**:
  - Stage 0: Setup & Frozen Classifier Wrap
  - Stage 1: Corpus Acquisition (4 repos, 7,704 chunks, SQLite + JSONL)
  - Stage 2: Hybrid Retrieval Engine (BM25 + BGE Dense + RRF + Cross-Encoder Reranker)
  - Stage 3: Retrieval Evaluation & 4-Way Ablation (51 verified ground-truth pairs)

---

## 2. Stage 2 & 3 Deliverables & Verification

### A. Hybrid Retrieval Engine (`src/retrieval/`)
1. **BM25 (`src/retrieval/bm25.py`)**: Code-aware tokenizer preserving code identifiers, CamelCase, snake_case, versions (e.g. `v2.0`), paths, and URLs.
2. **Dense (`src/retrieval/dense.py`)**: `BAAI/bge-base-en-v1.5` with BGE query instruction prefix (`"Represent this sentence for searching relevant passages: "`), disk-cached embeddings in `data/processed/embeddings_cache.npz` (7,689 vectors), and exact numpy cosine similarity.
3. **Fusion (`src/retrieval/fusion.py`)**: Reciprocal Rank Fusion (RRF, $k=60$) deduplicating by `chunk_id` and tracking per-method ranks.
4. **Reranker (`src/retrieval/rerank.py`)**: `BAAI/bge-reranker-base` cross-encoder reranking fused candidate pools and assigning stable IDs (`E1`, `E2`, ...).
5. **Coordinator (`src/retrieval/index.py`)**: `HybridRetriever` supporting `"bm25"`, `"dense"`, `"hybrid"`, and `"hybrid_rerank"`.
6. **CLI Search Acceptance**: `python -m src.cli search "<query>" --repo <repo>` verified across sparse and hybrid reranking pipelines.

### B. Retrieval Ground Truth (`eval/build_retrieval_set.py`)
- Mined duplicate closures, `duplicate` labels, and explicit resolution cross-references across 4 repositories (`pallets/flask`, `encode/httpx`, `pydantic/pydantic`, `psf/requests`).
- Fetched 30 missing canonical issues on-demand via `fetch_single_issue` and added their chunks to the SQLite index so ground-truth targets are never missing.
- Assembled **51 verified query-target pairs** in `eval/retrieval_ground_truth.json`:
  - `encode/httpx`: 15 pairs
  - `pallets/flask`: 7 pairs
  - `psf/requests`: 23 pairs
  - `pydantic/pydantic`: 6 pairs

### C. 4-Way Retrieval Ablation (`eval/run_retrieval_eval.py`)
Evaluated across all 51 queries with strict temporal cutoff filtering (excluding the query issue itself and all chunks created after the query timestamp):

| Retrieval Method | Recall@5 | Precision@5 | MRR@10 | nDCG@10 | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. BM25 Only** | **0.6471** | 0.1294 | 0.4877 | 0.5358 | **63.09 ms** |
| **2. BGE Dense Only** | **0.6667** | 0.1333 | 0.5042 | 0.5662 | 279.86 ms |
| **3. BM25 + BGE (RRF $k=60$)** | 0.6078 | 0.1216 | **0.5313** | **0.5939** | 135.00 ms |
| **4. RRF + BGE Reranker** | 0.5098 | 0.1020 | 0.3592 | 0.4273 | 1,141.81 ms |

**Key Findings**:
- **BGE Dense Only** achieves the highest individual Recall@5 (**66.67%** vs 64.71% for BM25).
- **BM25 + BGE (RRF $k=60$)** achieves the highest ranking quality (**MRR@10: 0.5313, nDCG@10: 0.5939**), demonstrating that multi-stage fusion concentrates true canonical duplicates into the top 1–2 positions.
- In `pallets/flask` and `psf/requests`, dense semantic retrieval significantly outperforms BM25 (Flask: 85.71% vs 71.43%; Requests: 73.91% vs 65.22%).
- Full report committed to `eval/reports/retrieval_ablation.md` and `eval/reports/retrieval_ablation.json`.

---

## 3. Test Suite Status

- **Automated Unit Tests**: 60 passed in 0.55s (`tests/test_retrieval.py`, `tests/test_chunker.py`, `tests/test_ingest.py`, `tests/test_classifier.py`, `tests/test_cli.py`, `tests/test_preprocessing.py`, `tests/test_evaluation.py`).

---

## 4. Next Step
Proceed to **Stage 4: Generation & Structured LLM Synthesis** (Ollama Qwen 4B JSON analysis with evidence citation IDs) and **Stage 5: NLI Guardrail**.
