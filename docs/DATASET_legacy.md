# NLBSE 2024 Issue Report Classification Dataset

## 1. Overview and Source

The dataset used in this microproject is the official **NLBSE'24 Tool Competition on Issue Report Classification** benchmark dataset.

- **Official Repository**: [https://github.com/nlbse2024/issue-report-classification](https://github.com/nlbse2024/issue-report-classification)
- **Competition Website**: [https://nlbse2024.github.io/tools/](https://nlbse2024.github.io/tools/)
- **Total Records**: 3,000 labelled GitHub issue reports
- **Splits**:
  - Training: exactly 1500 records (`issues_train.csv`)
  - Testing: exactly 1500 records (`issues_test.csv`)

## 2. Citation

```bibtex
@inproceedings{nlbse2024,
  author={Kallis, Rafael and Colavito, Giuseppe and Al-Kaswan, Ali and Pascarella, Luca and Chaparro, Oscar and Rani, Pooja},
  title={The NLBSE'24 Tool Competition},
  booktitle={Proceedings of The 3rd International Workshop on Natural Language-based Software Engineering (NLBSE'24)},
  year={2024}
}

@article{kallis2020tickettagger,
  author={Kallis, Rafael and Di Sorbo, Andrea and Canfora, Gerardo and Panichella, Sebastiano},
  title={Predicting issue types on GitHub},
  journal={Science of Computer Programming},
  volume={205},
  pages={102598},
  year={2021},
  doi={10.1016/j.scico.2020.102598}
}
```

## 3. Licensing and Attribution Notice

> [!WARNING]
> **Redistribution Uncertainty**: The upstream official repository `nlbse2024/issue-report-classification` currently provides an empty `LICENSE` file (0 bytes). While the underlying issue reports originate from public open-source projects on GitHub, the redistribution permissions of the compiled CSVs are not explicitly licensed. Therefore:
> - The raw dataset is strictly stored under `data/external/nlbse2024/` and excluded from Git commits via `.gitignore`.
> - Code, evaluation results, and preprocessing pipelines are tracked.

## 4. Dataset Fields

The raw CSV files contain the following five fields:

| Field | Type | Description |
|---|---|---|
| `repo` | `string` | The GitHub repository identifier (`owner/repo`) |
| `created_at` | `string` | Timestamp when the issue was created on GitHub |
| `label` | `string` | Canonical ground-truth classification label |
| `title` | `string` | Issue report title |
| `body` | `string` | Full Markdown text of the issue report body |

## 5. Classes and Repository Distribution

The dataset covers **5 diverse open-source repositories** and **3 mutually exclusive classes**:
- `bug`
- `feature`
- `question`

Both training and test sets are **perfectly balanced**, containing exactly **100 records per repository per class**:

| Repository | Split | Bug | Feature | Question | Total |
|---|---|---|---|---|---|
| `facebook/react` | Train / Test | 100 / 100 | 100 / 100 | 100 / 100 | 300 / 300 |
| `tensorflow/tensorflow` | Train / Test | 100 / 100 | 100 / 100 | 100 / 100 | 300 / 300 |
| `microsoft/vscode` | Train / Test | 100 / 100 | 100 / 100 | 100 / 100 | 300 / 300 |
| `bitcoin/bitcoin` | Train / Test | 100 / 100 | 100 / 100 | 100 / 100 | 300 / 300 |
| `opencv/opencv` | Train / Test | 100 / 100 | 100 / 100 | 100 / 100 | 300 / 300 |
| **Total** | **All** | **500 / 500** | **500 / 500** | **500 / 500** | **1500 / 1500** |

## 6. Text Length Distributions

### Training Split (1,500 records)
- **Title Words**: min=1, max=39, mean=7.9, median=7.0
- **Body Words**: min=1, max=21586, mean=228.7, median=142.0
- **Combined Text Words**: min=2, max=21595, mean=236.6, median=150.0
- **Combined Characters**: min=3, max=202447, mean=2353.3, median=1141.5

### Test Split (1,500 records)
- **Title Words**: min=1, max=50, mean=8.0, median=7.0
- **Body Words**: min=0, max=5048, mean=216.2, median=144.0
- **Combined Text Words**: min=1, max=5050, mean=224.2, median=151.0
- **Combined Characters**: min=3, max=61627, mean=2200.4, median=1165.0

## 7. Data Quality & Overlap Checks

| Check | Result |
|---|---|
| Train records with missing title | 0 |
| Train records with empty title | 0 |
| Train records with empty body | 0 |
| Test records with missing title | 0 |
| Test records with empty title | 0 |
| Test records with empty body | 2 |
| Exact train/test duplicates | 3 |
| Same repository & title across train and test | 14 |

## 8. Dataset Limitations

1. **Synthetic balance**: The 1:1:1 balance does not reflect real-world repository distributions where bugs usually outnumber questions or features.
2. **Extreme text length variance**: Issue bodies range from short one-liners to >100,000 character logs/stack traces. Truncation strategy must be carefully handled for transformer architectures.
3. **Repository-specific conventions**: Each project uses distinct issue templates (e.g., React's checklist vs. TensorFlow's environment dump), creating strong domain style signals.
