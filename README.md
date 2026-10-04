# AFLORA-MEC-Reproducible
Reproducible implementation of AFLORA – Adaptive Federated Learning for Cloud-Edge Resource Optimization in Mobile Edge Computing (CEDN + DQN + O2A)
# AFLORA: Adaptive Federated Learning Framework for Cloud-Edge Resource Optimization in Mobile Edge Computing

This repository provides a **reproducible, Kaggle-ready implementation** of the core ideas of AFLORA.

AFLORA combines three main components:
- **CEDN** (Capsule-inspired Encoder) for distribution-aware feature extraction under Non-IID data
- **DQN** for adaptive task offloading decisions (Local / Edge / Cloud)
- **O2A** (Osprey-inspired optimization) for power and bandwidth allocation

The goal is to reduce communication cost while maintaining model performance in Mobile Edge Computing (MEC) environments with Non-IID data.

> **Important Note**  
> This is a simplified and publicly reproducible version of the framework.  
> It is intended for transparency and verification of the overall pipeline.  
> The exact numerical results reported in the original paper (e.g., 82.4% communication cost reduction) were obtained with the full experimental configuration and longer training.

---

## Features

- Dirichlet-based Non-IID partitioning of CIFAR-10
- Lightweight CEDN-style feature extractor
- Deep Q-Network for offloading decisions
- Simplified Osprey-inspired resource allocation (power + bandwidth)
- Communication cost simulation under Low Communication Power (LCP) and High Communication Power (HCP)
- Multi-seed evaluation
- Comparison against a simple FedAvg baseline
- Fully runnable on Kaggle (GPU recommended)

---

## Quick Start (Kaggle – Recommended)

1. Create a new Kaggle Notebook
2. Upload `AFLORA_Kaggle_Improved.ipynb`
3. Set **Accelerator → GPU**
4. Click **Run All**

After the run finishes you will get:
- `aflora_results.png` – Accuracy and communication cost curves
- `aflora_summary.csv` – Summary metrics

---

## Local Installation

```bash
pip install -r requirements.txt
jupyter notebook AFLORA_Kaggle_Improved.ipynb
