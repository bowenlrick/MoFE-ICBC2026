# MoFE-ICBC2026

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![IEEE Xplore](https://img.shields.io/badge/IEEE-Published-00629B.svg)](https://ieeexplore.ieee.org/document/11575439)
[![arXiv](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)](#) 

Official implementation of the **IEEE ICBC 2026** paper:  
**"MoFE: A Novel Mixture-of-Experts Framework with Fourier Neural Operators for Cryptocurrency Forecasting"**

🔗 **Paper Links:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11575439) | [arXiv (Pending)](#)

---

## 📖 Introduction

Forecasting cryptocurrency prices is challenging due to inherent non-stationarity, abrupt regime shifts, and multi-scale stochastic dependencies. We propose **MoFE**, a novel deep learning framework that integrates **Fourier Neural Operators (FNOs)** within a **Mixture-of-Experts (MoE)** architecture. 

By modeling cryptocurrency volatility as a superposition of multi-frequency components, MoFE effectively mitigates the phase-lag effect common in time-series forecasting, achieving State-of-the-Art (SOTA) performance in directional accuracy and risk-adjusted returns (Sharpe Ratio).

## 🚀 Model Architecture

<img width="2458" height="776" alt="MoFE Architecture" src="https://github.com/user-attachments/assets/573acc12-c87e-4281-b82b-c21da4655770" />

## 📁 Repository Structure

```text
MoFE-ICBC2026/
├── data/
│   ├── train.csv         # Training set (chronologically split)
│   ├── val.csv           # Validation set
│   └── test.csv          # Out-of-sample testing set
├── train_t1.py           # Training & evaluation script for T+1 horizon
├── train_t5.py           # Training & evaluation script for T+5 horizon
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## 🛠️ Installation & Setup

1. Clone this repository:
```bash
git clone [https://github.com/bowenlrick/MoFE-ICBC2026.git](https://github.com/bowenlrick/MoFE-ICBC2026.git)
cd MoFE-ICBC2026
```

2. Create a virtual environment and install dependencies:
```bash
conda create -n mofe_env python=3.9
conda activate mofe_env
pip install -r requirements.txt
```

## 📊 Dataset

The pre-processed daily Bitcoin OHLCV dataset is included in the `data/` directory. The feature space has been expanded to an 8-dimensional vector, including logarithmic returns, RSI, MACD, Relative Volume, and a Volatility Proxy. The data is strictly split chronologically to prevent data leakage.

## 💻 Usage

We provide separate training scripts for different forecasting horizons ($T+1$ and $T+5$). The scripts will automatically train the MoFE model, evaluate it on the test set, and generate visualization plots.

**For 1-step ahead forecasting (T+1):**
```bash
python train_t1.py
```

**For 5-step ahead forecasting (T+5):**
```bash
python train_t5.py
```

*Note: The best model checkpoints will be saved locally, and the comprehensive performance metrics (RMSE, MAE, R2, IC, DA, Sharpe Ratio) will be printed in the console.*

## 🏆 Main Results

MoFE achieves SOTA performance across multiple horizons. Below is a snapshot of our simulated trading performance on the test set:

| Horizon | RMSE ($) | R² | IC | Directional Acc. | Sharpe Ratio | Cumulative ROI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T+1** | 1582.64 | 0.9827 | 0.1957 | **58.43%** | **3.89** | **4.41x** |
| **T+5** | 3360.33 | 0.9186 | 0.1389 | **56.10%** | **1.73** | **4.91x** |

## 📝 Citation

If you find this repository or our paper useful for your research, please consider citing our work:

```bibtex
@inproceedings{liu2026mofe,
  title={MoFE: A Novel Mixture-of-Experts Framework with Fourier Neural Operators for Cryptocurrency Forecasting},
  author={Liu, Bowen and Sun, Mingming},
  booktitle={2026 IEEE International Conference on Blockchain and Cryptocurrency (ICBC)},
  year={2026},
  organization={IEEE},
  doi={10.1109/ICBC67748.2026.11575439},
  url={[https://ieeexplore.ieee.org/document/11575439](https://ieeexplore.ieee.org/document/11575439)}
}
```

## 📜 License

This project is released under the [MIT License](LICENSE).

## 📧 Contact

If you have any questions, bug reports, or feature requests, please feel free to open an **Issue** on GitHub. 

For academic collaborations or further discussions, you can reach out to the author:
- **Bowen Liu**: `bliu59@u.rochester.edu` (Academic) or `bowenl.rick@gmail.com` (Personal)
