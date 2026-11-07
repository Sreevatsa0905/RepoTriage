# RepoTriage Automated Synthetic Perturbation Test Report
**Timestamp**: 2026-11-07T11:05:46.082050+00:00  
**Caveat**: *Synthetic perturbation test; labels are known by construction; claims are easier and more templated than real LLM output; not a substitute for human review.*  
**Sample**: 300 rule-constructed pairs (~100 per class)  
**Runtime**: 22.75 s  

---

## 1. Performance at Production Thresholds (tau_s=0.80, tau_c=0.80)
- **Accuracy**: **88.67%**
- **Macro-F1**: **0.8862**

### Per-Class Breakdown (Production Thresholds)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **SUPPORTED** | 0.9406 | 0.9500 | 0.9453 | 100 |
| **CONTRADICTED** | 0.8142 | 0.9200 | 0.8638 | 100 |
| **INSUFFICIENT_EVIDENCE** | 0.9186 | 0.7900 | 0.8495 | 100 |

### Confusion Matrix (Rows: Ground Truth, Columns: Predicted)

| True \ Pred | SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE |
| :--- | :---: | :---: | :---: |
| **SUPPORTED** | 95 | 0 | 5 |
| **CONTRADICTED** | 6 | 92 | 2 |
| **INSUFFICIENT_EVIDENCE** | 0 | 21 | 79 |

---

## 2. Threshold Calibration Sweep

| tau_s | tau_c | Accuracy | Macro-F1 | Supported Pass Rate | NEI False-Support Rate | Contradicted False-Support Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.30 | 0.50 | 87.33% | 0.8718 | 97.00% | 1.00% | 8.00% |
| 0.30 | 0.70 | 87.33% | 0.8718 | 97.00% | 1.00% | 8.00% |
| 0.30 | 0.80 | 89.00% | 0.8892 | 97.00% | 1.00% | 8.00% |
| 0.30 | 0.90 | 90.00% | 0.8995 | 97.00% | 1.00% | 8.00% |
| 0.50 | 0.50 | 87.33% | 0.8722 | 96.00% | 0.00% | 8.00% |
| 0.50 | 0.70 | 87.33% | 0.8722 | 96.00% | 0.00% | 8.00% |
| 0.50 | 0.80 | 89.00% | 0.8895 | 96.00% | 0.00% | 8.00% |
| 0.50 | 0.90 | 90.00% | 0.8997 | 96.00% | 0.00% | 8.00% |
| 0.70 | 0.50 | 87.33% | 0.8722 | 96.00% | 0.00% | 7.00% |
| 0.70 | 0.70 | 87.33% | 0.8722 | 96.00% | 0.00% | 7.00% |
| 0.70 | 0.80 | 89.00% | 0.8895 | 96.00% | 0.00% | 7.00% |
| 0.70 | 0.90 | 90.00% | 0.8997 | 96.00% | 0.00% | 7.00% |
| 0.80 | 0.50 | 87.00% | 0.8690 | 95.00% | 0.00% | 6.00% |
| 0.80 | 0.70 | 87.00% | 0.8690 | 95.00% | 0.00% | 6.00% |
| 0.80 | 0.80 | 88.67% | 0.8862 | 95.00% | 0.00% | 6.00% |
| 0.80 | 0.90 | 89.67% | 0.8964 | 95.00% | 0.00% | 6.00% |
| 0.90 | 0.50 | 86.67% | 0.8657 | 94.00% | 0.00% | 6.00% |
| 0.90 | 0.70 | 86.67% | 0.8657 | 94.00% | 0.00% | 6.00% |
| 0.90 | 0.80 | 88.33% | 0.8829 | 94.00% | 0.00% | 6.00% |
| 0.90 | 0.90 | 89.33% | 0.8931 | 94.00% | 0.00% | 6.00% |