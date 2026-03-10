# Stochastic PAC-Bayesian Transformers

**Paper:** *Stochastic PAC-Bayesian Transformers for Network Intrusion Detection and Natural Language Processing Applications*

**Authors:** Roger Nick Anaedevha, Alexander G. Trofimov, Yuri V. Borodachev
**Affiliation:** National Research Nuclear University MEPhI, Moscow, Russia

## Overview

This repository provides the complete implementation for reproducing the results of the paper. The framework introduces stochastic PAC-Bayesian transformers that unify calibrated uncertainty quantification, adversarial robustness, and label efficiency within a single architecture.

**Key results across 9 datasets in 3 security domains:**
- **96.8 +/- 0.8%** accuracy with ECE of **0.043 +/- 0.006**
- **88.3 +/- 1.5%** adversarial robustness retention under FGSM, PGD, C&W, and TextFooler
- **68%** reduction in labeling requirements via uncertainty-guided active learning

## Architecture

The model replaces deterministic attention projections with learned Gaussian distributions, enabling:
1. **Bayesian Attention** at layers {3, 6, 9, 12} with variational Q, K, V projections
2. **Variational Embeddings** propagating input-level uncertainty
3. **MC Sampling** (30 train / 50 inference) for uncertainty quantification
4. **EOT Adversarial Training** requiring robustness across stochastic instantiations
5. **PAC-Bayesian KL regularization** providing joint calibration-robustness bounds

## Installation

```bash
pip install -r requirements.txt
```

### Hardware Requirements

| | Minimum | Recommended |
|---|---------|-------------|
| GPU | NVIDIA P100 (16 GB) | NVIDIA V100/A100 (32+ GB) |
| RAM | 32 GB | 128 GB |
| Storage | 50 GB | 100 GB SSD |

## Datasets

**Network Intrusion Detection (tabular):**
- CIC-IoT-2023, CSE-CICIDS2018, UNSW-NB15
- Download: https://doi.org/10.34740/KAGGLE/DSV/12479689

**Toxic Content Detection (text):**
- MetaHate, HatEval, Founta (standard academic repositories)

**Fake News Detection (text):**
- LIAR, FakeNewsNet, ISOT (standard academic repositories)

## Quick Start

### Training

```bash
# Network intrusion detection
python scripts/train.py \
    --domain network \
    --dataset cic_iot_2023 \
    --data_path data/cic_iot_2023.csv \
    --config configs/default_config.yaml

# Toxic content detection
python scripts/train.py \
    --domain toxic \
    --dataset metahate \
    --data_path data/metahate.csv \
    --config configs/default_config.yaml

# Fake news detection
python scripts/train.py \
    --domain fakenews \
    --dataset liar \
    --data_path data/liar.csv \
    --config configs/default_config.yaml
```

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/cic_iot_2023/fold_0/best_model.pt \
    --domain network \
    --dataset cic_iot_2023 \
    --data_path data/cic_iot_2023.csv
```

### Full Experimental Suite

Reproduces all results across 9 datasets, 5 folds, 3 seeds:

```bash
bash scripts/run_experiments.sh /path/to/data /path/to/output
```

## Project Structure

```
stochastic_pac_bayesian_transformers/
├── configs/
│   └── default_config.yaml          # All hyperparameters (Table F.1)
├── src/
│   ├── models/
│   │   ├── bayesian_attention.py    # Variational Q,K,V projections (Eq. 5)
│   │   ├── variational_embedding.py # Domain-specific embeddings (Eq. A.1-A.3)
│   │   ├── stochastic_transformer.py# Full model with MC sampling
│   │   └── positional_encoding.py   # Sinusoidal positional encoding
│   ├── attacks/
│   │   ├── fgsm.py                  # FGSM attack (Algorithm F.1)
│   │   ├── pgd.py                   # PGD attack (Algorithm F.2)
│   │   ├── cw.py                    # Carlini-Wagner L2 attack
│   │   └── eot.py                   # EOT wrapper (Eq. 8)
│   ├── training/
│   │   ├── losses.py                # 5-component loss (Eq. 9)
│   │   ├── trainer.py               # Full training loop (Algorithm 2)
│   │   └── active_learning.py       # Uncertainty-guided acquisition
│   ├── evaluation/
│   │   ├── uncertainty.py           # Epistemic/aleatoric decomposition
│   │   ├── calibration.py           # ECE, MCE, Brier, AUROC
│   │   └── robustness.py            # Multi-attack robustness evaluation
│   └── data/
│       ├── network_datasets.py      # CIC-IoT, CICIDS, UNSW loaders
│       ├── toxic_datasets.py        # MetaHate, HatEval, Founta loaders
│       └── fake_news_datasets.py    # LIAR, FakeNewsNet, ISOT loaders
├── scripts/
│   ├── train.py                     # Training entry point
│   ├── evaluate.py                  # Evaluation entry point
│   └── run_experiments.sh           # Full experimental pipeline
├── notebooks/
│   └── demo.ipynb                   # Interactive demonstration
├── requirements.txt
└── README.md
```

## Hyperparameters

All hyperparameters match Table F.1 of the supplementary material:

| Parameter | Value |
|-----------|-------|
| Transformer layers | 12 |
| Attention heads | 8 |
| Hidden dimension | 256 |
| Feed-forward dimension | 1024 |
| Bayesian layers | {3, 6, 9, 12} |
| MC samples (train/inference) | 30/50 |
| Loss weights (KL/cal/adv/reg) | 0.01/0.05/0.2/0.001 |
| Learning rate | 2e-5 (network), 5e-5 (text) |
| Batch size (effective) | 128 (32 x 4 accumulation) |
| Optimizer | AdamW (beta1=0.9, beta2=0.999) |

## Reproducibility

```python
# Set all seeds (Appendix J)
import torch, numpy as np, random, os
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

## Citation

If you use this code, please cite:

```bibtex
@article{anaedevha2025stochastic,
  title={Stochastic {PAC}-{B}ayesian Transformers for Network Intrusion Detection
         and Natural Language Processing Applications},
  author={Anaedevha, Roger Nick and Trofimov, Alexander G. and Borodachev, Yuri V.},
  journal={Knowledge and Information Systems},
  year={2025}
}
```

## License

MIT License. See the manuscript repository for details.
