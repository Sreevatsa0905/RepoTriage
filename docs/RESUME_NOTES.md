# Technical Resume Draft & Experience Notes

### Experience / Project Bullet Points

> **Local-First GitHub Issue Intelligence & Resolution Copilot (RepoTriage)**
> * Built an offline-first issue resolution copilot combining a selective TF-IDF + Logistic Regression classifier (73.58% coverage at 0.50 confidence, achieving 82.05% accuracy on covered issues), hybrid BM25 + BGE dense retrieval (Dense Recall@5: 0.8039 [95% CI: 0.6863, 0.9020]; RRF MRR@10: 0.6018 [95% CI: 0.4948, 0.7139] on $n=51$ duplicate query pairs), local Qwen 4B synthesis, and DeBERTa-v3 natural language inference (NLI) claim verification.
> * Discovered that an off-the-shelf cross-encoder reranker (`bge-reranker-base`) degraded duplicate-issue retrieval Recall@5 from 0.7647 (RRF) and 0.8039 (Dense) down to 0.5490 (pure reranking $\alpha=\text{None}$ / $\alpha=1.0$), resolved via convex reciprocal-rank score blending ($\alpha = 0.50$, recovering 0.7843 Recall@5 and 0.6022 MRR@10).
> * Implemented multi-premise sentence-window NLI verification enforcing strict dominance thresholds ($\tau_s = 0.80, \tau_c = 0.80$), achieving 0.00% (0 / 80) false-support on hard negative controls, 1.25% (1 / 80) on easy negative controls, and 95.00% (95 / 100) supported recall on synthetic rule-perturbed developer claims.
> * Engineered deterministic sequential GPU memory management for consumer hardware (NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM), sequentially loading and unloading PyTorch embedding/reranking models before Ollama inference to guarantee zero-OOM execution across pipeline stages.
> * Built an asynchronous FastAPI REST service with single-flight concurrency locking (HTTP 429) alongside a Vite React + TypeScript dark-theme copilot displaying unmerged dual classification and interactive claim verification traces.
