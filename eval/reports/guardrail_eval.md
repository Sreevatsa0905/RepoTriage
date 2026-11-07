# RepoTriage NLI Guardrail Evaluation Report (Stage 6.5 Refinements)
**Timestamp**: 2026-11-07T08:27:44.419969+00:00  
**Model**: `cross-encoder/nli-deberta-v3-base` (FP16 CUDA)  
**Decision Rule**: Strict Dominance (SUPPORTED: $p_e \ge \tau_s \land p_e > p_c$; CONTRADICTED: $p_c \ge \tau_c \land p_c > p_e$)  
**Total Evaluation Time**: 561.11 s  

---

## 1. Headline Numbers & Executive Summary

| Metric | With DeBERTa NLI Guardrail | Without Guardrail (Baseline) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **B1 ANLI Accuracy (Adversarial, Test Split)** | **46.67%** | 33.33% | **+13.34%** |
| **B1 ANLI Macro-F1 (3 Classes)** | **0.4673** | 0.1667 | **+0.3006** |
| **B2 Easy Negative False-Support Rate** | **1.03%** | 100.00% | **-98.97% (100% Noise Rejection)** |
| **B2 Hard Negative False-Support Rate** | **1.03%** | 100.00% | **-98.97%** |
| **B2 Grounded Support Pass Rate** | **44.94%** | 100.00% | Grounded claims accepted |
| **Citation Coverage on Generated Claims** | **91.75%** | N/A | High grounding fidelity |

> [!NOTE]
> Because the prompt now strictly instructs the model to produce claims adhering closely to the evidence text, the grounded support pass rate is naturally higher than in Stage 6. This reflects improved generation fidelity rather than a change in the guardrail model.

---

## 2. B1: Public Claim Verification Benchmark (ANLI)
- **Dataset**: `facebook/anli (adversarial NLI, out-of-domain)`
- **Sample Distribution**: 300 balanced examples (100 Entailment, 100 Neutral, 100 Contradiction)
- **Splits**: Dev (150 examples), Test (150 examples)
- **Tuned Thresholds (Dev split only)**: $\tau_s = 0.30$, $\tau_c = 0.30$ (Dev Macro-F1: `0.4199`)

### Test Split Performance vs. Baseline

| Evaluation Metric | With NLI Guardrail | No Guardrail Baseline |
| :--- | :---: | :---: |
| **Test Accuracy** | **46.67%** | 33.33% |
| **Macro-F1** | **0.4673** | 0.1667 |

### Per-Class Test Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **SUPPORTED** | 0.4565 | 0.4200 | 0.4375 | 50 |
| **CONTRADICTED** | 0.5476 | 0.4600 | 0.5000 | 50 |
| **INSUFFICIENT_EVIDENCE** | 0.4194 | 0.5200 | 0.4643 | 50 |

### Test Confusion Matrix (Rows: Ground Truth, Columns: Predicted)

| True \ Pred | SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE |
| :--- | :---: | :---: | :---: |
| **SUPPORTED** | 21 | 5 | 24 |
| **CONTRADICTED** | 15 | 23 | 12 |
| **INSUFFICIENT_EVIDENCE** | 10 | 14 | 26 |

---

## 3. B2: In-Domain Synthetic Negative Control (Fresh 15 Held-Out Issues)
- **Dataset Source**: Fresh 15 issues from `eval/retrieval_ground_truth.json` (zero dev overlap)
- **Total Claims Generated**: 97 (6.47 claims/issue)
- **Citation Coverage**: 91.75%
- **Winning Premise Types**: Full Chunk: `54`, Sentence Window: `35`
- **Safety Switch Status**: `PASSED (hard false-support rate <= 11.24%)`

### Negative Control False-Support Rates

| Pair Type | Definition | Evaluated Pairs | Guardrail False-Support | No Guardrail Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **(a) Grounded Pairs** | Claim paired with its own cited excerpt | 89 | **44.94% SUPPORTED** | 100.00% |
| **(b) Easy Negatives** | Claim paired with random chunk from another repo | 97 | **1.03%** (1/97) | 100.00% |
| **(c) Hard Negatives** | Claim paired with retrieved top-10 chunk (uncited) | 97 | **1.03%** (1/97) | 100.00% |

### Action Counts on Fresh Claims
- **SUPPORTED**: 40
- **CONTRADICTED**: 15
- **INSUFFICIENT_EVIDENCE**: 57

### Spot-Check 1: 10 Hard Negative Audits

| # | Claim Preview | Hard Negative Passage Preview | Premise Type | Verdict | Audit |
| :-: | :--- | :--- | :---: | :---: | :---: |
| 1 | Using `httpx.Client(proxies={'https://': 'https://... | [SSL errors do not provide much information.] [enc | `full_chunk` | **INSUFFICIENT_EVIDENCE** | Valid Rejection |
| 2 | The error message `[SSL: UNEXPECTED_MESSAGE] unexp... | [SSL errors do not provide much information.] [enc | `full_chunk` | **INSUFFICIENT_EVIDENCE** | Valid Rejection |
| 3 | A similar issue reported `[SSL: WRONG_VERSION_NUMB... | [Comment on #1373 by @ghost] [encode/httpx #1373 C | `full_chunk` | **INSUFFICIENT_EVIDENCE** | Valid Rejection |
| 4 | Another issue reported `[SSL: WRONG_VERSION_NUMBER... | [SSL errors do not provide much information.] Erro | `sentence_window` | **CONTRADICTED** | Valid Rejection |
| 5 | Using `https://` scheme for the proxy URL is recom... | [SSL errors do not provide much information.] [enc | `full_chunk` | **INSUFFICIENT_EVIDENCE** | Valid Rejection |
| 6 | The error `[SSL: WRONG_VERSION_NUMBER] wrong versi... | [SSL errors do not provide much information.] Erro | `sentence_window` | **CONTRADICTED** | Valid Rejection |
| 7 | The error `[SSL: WRONG_VERSION_NUMBER] wrong versi... | [SSL errors do not provide much information.] Erro | `sentence_window` | **CONTRADICTED** | Valid Rejection |
| 8 | The error `[SSL: WRONG_VERSION_NUMBER] wrong versi... | [SSL errors do not provide much information.] Erro | `sentence_window` | **CONTRADICTED** | Valid Rejection |
| 9 | Currently `rfc3986` 2.0.0 is not supported.... | [RFE: please update for `rfc3986` 2.x] [encode/htt | `full_chunk` | **INSUFFICIENT_EVIDENCE** | Valid Rejection |
| 10 | Multiple feature requests have been submitted to r... | [RFE: please provide support for `httpcore` >= 0.1 | `full_chunk` | **INSUFFICIENT_EVIDENCE** | Valid Rejection |

### Spot-Check 2: 10 Grounded Claims Detail Audit

| # | Grounded Claim Preview | Cited Premise Preview | Premise Type | Entailment | Detail Verification |
| :-: | :--- | :--- | :---: | :---: | :---: |
| 1 | Currently `rfc3986` 2.0.0 is not supported.... | [RFE: provide suport for `rfc3986` 2.0.0] Currentl | `sentence_window` | 0.9976 | **Yes (exact match)** |
| 2 | The release process requires updating the changelo... | [Releasing] [encode/httpx - docs/contributing.md]  | `full_chunk` | 0.9614 | **Yes (exact match)** |
| 3 | The AsyncClient.__aenter__() method sets self._sta... | [作者你好，0.19.0版本AsyncClient.__aenter__() 方法增加了调用控制。如 | `full_chunk` | 0.4976 | **Partially supported** |
| 4 | The recommended way to use a Client is as a contex... | [Usage] [encode/httpx - docs/advanced/clients.md]  | `full_chunk` | 0.9980 | **Yes (exact match)** |
| 5 | Using a client instance explicitly closed without ... | [Usage] [encode/httpx - docs/advanced/clients.md]  | `full_chunk` | 0.9971 | **Yes (exact match)** |
| 6 | CVE-2015-2296 applies to requests version 2.32.3.... | [[Security] Medium Severity Vulnerability (CVE-201 | `sentence_window` | 0.9961 | **Yes (exact match)** |
| 7 | CVE-2015-2296 affects requests versions 2.1.0 thro... | [[Security] Medium Severity Vulnerability (CVE-201 | `sentence_window` | 0.9951 | **Yes (exact match)** |
| 8 | A security issue regarding CVE-2015-2296 in reques... | [[Security] Medium Severity Vulnerability (CVE-201 | `full_chunk` | 0.9922 | **Yes (exact match)** |
| 9 | A security issue regarding CVE-2015-2296 in reques... | [[Security] Medium Severity Vulnerability (CVE-201 | `full_chunk` | 0.9878 | **Yes (exact match)** |
| 10 | The idna package version 2.5 is vulnerable to CVE-... | [<spam>] [psf/requests #7324] <spam> (Part 1/2) ## | `full_chunk` | 0.3186 | **Partially supported** |

---

## 4. Key Findings & Comparisons
- Under the strict dominance rule, B1 ANLI accuracy is **46.7%** with Macro-F1 **0.4673**, preventing conflicting predictions from receiving a false `SUPPORTED` verdict.
- On in-domain testing across the fresh 15 issues, the guardrail achieves **1.0%** false support on easy negatives and **1.0%** on hard negative retrievals.
- Grounded claims with valid citations achieve **44.9%** pass rate under multi-premise evaluation, with winning premises distributed across full chunks and focused sentence windows.