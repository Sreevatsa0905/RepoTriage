# Retrieval Ablation Evaluation Report
**Timestamp**: 2026-11-07T05:34:47.521521+00:00  
**Queries Evaluated**: 51 verified duplicate/cross-reference pairs  

## Overall 4-Way Ablation Results

| Retrieval Method | Recall@5 | Precision@5 | MRR@10 | nDCG@10 | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. BM25 Only** | 0.6471 | 0.1294 | 0.4940 | 0.5449 | 86.57 ms |
| **2. BGE Dense Only** | 0.6863 | 0.1373 | 0.5167 | 0.5815 | 284.94 ms |
| **3. BM25 + BGE (RRF k=60)** | 0.6471 | 0.1294 | 0.5375 | 0.5997 | 133.18 ms |
| **4. RRF + BGE Reranker** | 0.3922 | 0.0784 | 0.2862 | 0.3437 | 442.23 ms |

## Per-Repository Breakdown (Recall@5 / MRR@10)

| Repository | Queries | BM25 Recall@5 | Dense Recall@5 | Hybrid Recall@5 | Rerank Recall@5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`encode/httpx`** | 15 | 0.5333 | 0.4000 | 0.4000 | 0.3333 |
| **`pallets/flask`** | 7 | 0.7143 | 0.8571 | 0.7143 | 0.4286 |
| **`psf/requests`** | 23 | 0.6522 | 0.7826 | 0.7391 | 0.4348 |
| **`pydantic/pydantic`** | 6 | 0.8333 | 0.8333 | 0.8333 | 0.3333 |
