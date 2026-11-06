# Retrieval Ablation Evaluation Report
**Timestamp**: 2026-11-06T18:30:18.595336+00:00  
**Queries Evaluated**: 51 verified duplicate/cross-reference pairs  

## Overall 4-Way Ablation Results

| Retrieval Method | Recall@5 | Precision@5 | MRR@10 | nDCG@10 | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. BM25 Only** | 0.6471 | 0.1294 | 0.4877 | 0.5358 | 91.84 ms |

## Per-Repository Breakdown (Recall@5 / MRR@10)

| Repository | Queries | BM25 Recall@5 | Dense Recall@5 | Hybrid Recall@5 | Rerank Recall@5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`encode/httpx`** | 15 | 0.5333 | 0.0000 | 0.0000 | 0.0000 |
| **`pallets/flask`** | 7 | 0.7143 | 0.0000 | 0.0000 | 0.0000 |
| **`psf/requests`** | 23 | 0.6522 | 0.0000 | 0.0000 | 0.0000 |
| **`pydantic/pydantic`** | 6 | 0.8333 | 0.0000 | 0.0000 | 0.0000 |
