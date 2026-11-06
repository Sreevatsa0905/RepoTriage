# Retrieval Ablation Evaluation Report
**Timestamp**: 2026-11-06T19:06:12.221974+00:00  
**Queries Evaluated**: 51 verified duplicate/cross-reference pairs  

## Overall 4-Way Ablation Results

| Retrieval Method | Recall@5 | Precision@5 | MRR@10 | nDCG@10 | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. BM25 Only** | 0.6471 | 0.1294 | 0.4877 | 0.5358 | 63.09 ms |
| **2. BGE Dense Only** | 0.6667 | 0.1333 | 0.5042 | 0.5662 | 279.86 ms |
| **3. BM25 + BGE (RRF k=60)** | 0.6078 | 0.1216 | 0.5313 | 0.5939 | 135.00 ms |
| **4. RRF + BGE Reranker** | 0.5098 | 0.1020 | 0.3592 | 0.4273 | 1141.81 ms |

## Per-Repository Breakdown (Recall@5 / MRR@10)

| Repository | Queries | BM25 Recall@5 | Dense Recall@5 | Hybrid Recall@5 | Rerank Recall@5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`encode/httpx`** | 15 | 0.5333 | 0.4000 | 0.2667 | 0.4000 |
| **`pallets/flask`** | 7 | 0.7143 | 0.8571 | 0.7143 | 0.5714 |
| **`psf/requests`** | 23 | 0.6522 | 0.7391 | 0.7391 | 0.5652 |
| **`pydantic/pydantic`** | 6 | 0.8333 | 0.8333 | 0.8333 | 0.5000 |
