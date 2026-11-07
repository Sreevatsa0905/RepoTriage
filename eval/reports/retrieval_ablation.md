# Retrieval Ablation Evaluation Report
**Timestamp**: 2026-11-07T12:32:25.884580+00:00  
**Queries Evaluated**: 51 verified duplicate/cross-reference pairs  
**Candidate Pool**: `issues only (source_types=['issue', 'comment'])` (docs excluded to eliminate candidate crowding)  
**Temporal & Self-Exclusion Leakage Check**: 100% Passed (query issues & post-query chunks strictly excluded)  

## 1. Headline 4-Way Ablation Results (Issues-Only Pool)

| Retrieval Method | Recall@5 | Precision@5 | MRR@10 | nDCG@10 | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. BM25 Only** | **0.7451** | 0.1490 | **0.5484** | **0.6017** | 111.67 ms |
| **2. BGE Dense Only** | **0.8039** | 0.1608 | **0.5730** | **0.6439** | 1348.81 ms |
| **3. BM25 + BGE (RRF k=60)** | **0.7647** | 0.1529 | **0.6018** | **0.6692** | 196.32 ms |
| **4. RRF + BGE Reranker (Blend α=0.50)** | **0.7843** | 0.1569 | **0.6022** | **0.6697** | 504.63 ms |

## 2. Before (Mixed Pool) vs. After (Issues-Only Pool) Comparison

| Configuration | Metric | BM25 Only | BGE Dense Only | BM25 + BGE (RRF k=60) | RRF + Reranker |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Before (Mixed Pool)** | Recall@5 | 0.6471 | 0.6863 | 0.6471 | 0.3922 |
| | MRR@10 | 0.4940 | 0.5167 | 0.5375 | 0.2862 |
| | nDCG@10 | 0.5449 | 0.5815 | 0.5997 | 0.3437 |
| **After (Issues-Only Pool)** | Recall@5 | **0.7451** | **0.8039** | **0.7647** | **0.7843** |
| | MRR@10 | **0.5484** | **0.5730** | **0.6018** | **0.6022** |
| | nDCG@10 | **0.6017** | **0.6439** | **0.6692** | **0.6697** |

## 3. Reranker Blending Sensitivity Analysis
*Formula: final = (1 - α) * rrf_minmax + α * sigmoid(rerank_logit). Headline fixed at α = 0.50 in advance.*

| Blending Strategy | Recall@5 | Precision@5 | MRR@10 | nDCG@10 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Reranker Only (α=None)** | 0.5490 | 0.1098 | 0.3956 | 0.4601 | 387.33 ms |
| **Blend α = 0.3** | 0.7843 | 0.1569 | 0.5812 | 0.6540 | 324.78 ms |
| **Blend α = 0.5** **(Headline Fixed)** | 0.7843 | 0.1569 | 0.6022 | 0.6697 | 319.51 ms |
| **Blend α = 0.7** | 0.7647 | 0.1529 | 0.5931 | 0.6629 | 403.14 ms |
| **Blend α = 1.0** | 0.5490 | 0.1098 | 0.3956 | 0.4601 | 414.94 ms |

## 4. Per-Repository Breakdown (Issues-Only Pool)

| Repository | Queries | BM25 Recall@5 | Dense Recall@5 | Hybrid Recall@5 | Rerank Recall@5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`encode/httpx`** | 15 | 0.7333 | 0.5333 | 0.6000 | 0.6000 |
| **`pallets/flask`** | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **`psf/requests`** | 23 | 0.6522 | 0.8696 | 0.7391 | 0.7826 |
| **`pydantic/pydantic`** | 6 | 0.8333 | 1.0000 | 1.0000 | 1.0000 |
