# RepoTriage Automated Synthetic Perturbation Test Report
**Timestamp**: 2026-11-07T09:34:52.386302+00:00  
**Caveat**: *Synthetic perturbation test; labels are known by construction; claims are easier and more templated than real LLM output; not a substitute for human review.*  
**Sample**: 300 rule-constructed pairs (~100 per class)  
**Runtime**: 21.30 s  

---

## 1. Performance at Production Thresholds (tau_s=0.80, tau_c=0.80)
- **Accuracy**: **51.67%**
- **Macro-F1**: **0.4623**

### Per-Class Breakdown (Production Thresholds)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **SUPPORTED** | 0.5625 | 0.0900 | 0.1552 | 100 |
| **CONTRADICTED** | 0.6838 | 0.8000 | 0.7373 | 100 |
| **INSUFFICIENT_EVIDENCE** | 0.3952 | 0.6600 | 0.4944 | 100 |

### Confusion Matrix (Rows: Ground Truth, Columns: Predicted)

| True \ Pred | SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE |
| :--- | :---: | :---: | :---: |
| **SUPPORTED** | 9 | 5 | 86 |
| **CONTRADICTED** | 5 | 80 | 15 |
| **INSUFFICIENT_EVIDENCE** | 2 | 32 | 66 |

---

## 2. Threshold Calibration Sweep

| tau_s | tau_c | Accuracy | Macro-F1 | Supported Pass Rate | NEI False-Support Rate | Contradicted False-Support Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.30 | 0.50 | 52.67% | 0.4782 | 12.00% | 3.00% | 9.00% |
| 0.30 | 0.70 | 52.33% | 0.4757 | 12.00% | 3.00% | 9.00% |
| 0.30 | 0.80 | 52.33% | 0.4776 | 12.00% | 3.00% | 9.00% |
| 0.30 | 0.90 | 52.67% | 0.4811 | 12.00% | 3.00% | 9.00% |
| 0.50 | 0.50 | 52.67% | 0.4751 | 11.00% | 2.00% | 9.00% |
| 0.50 | 0.70 | 52.33% | 0.4726 | 11.00% | 2.00% | 9.00% |
| 0.50 | 0.80 | 52.33% | 0.4745 | 11.00% | 2.00% | 9.00% |
| 0.50 | 0.90 | 52.67% | 0.4779 | 11.00% | 2.00% | 9.00% |
| 0.70 | 0.50 | 52.00% | 0.4635 | 9.00% | 2.00% | 8.00% |
| 0.70 | 0.70 | 51.67% | 0.4609 | 9.00% | 2.00% | 8.00% |
| 0.70 | 0.80 | 51.67% | 0.4629 | 9.00% | 2.00% | 8.00% |
| 0.70 | 0.90 | 52.00% | 0.4663 | 9.00% | 2.00% | 8.00% |
| 0.80 | 0.50 | 52.00% | 0.4629 | 9.00% | 2.00% | 5.00% |
| 0.80 | 0.70 | 51.67% | 0.4603 | 9.00% | 2.00% | 5.00% |
| 0.80 | 0.80 | 51.67% | 0.4623 | 9.00% | 2.00% | 5.00% |
| 0.80 | 0.90 | 52.00% | 0.4657 | 9.00% | 2.00% | 5.00% |
| 0.90 | 0.50 | 51.67% | 0.4528 | 7.00% | 1.00% | 4.00% |
| 0.90 | 0.70 | 51.33% | 0.4503 | 7.00% | 1.00% | 4.00% |
| 0.90 | 0.80 | 51.33% | 0.4523 | 7.00% | 1.00% | 4.00% |
| 0.90 | 0.90 | 51.67% | 0.4557 | 7.00% | 1.00% | 4.00% |