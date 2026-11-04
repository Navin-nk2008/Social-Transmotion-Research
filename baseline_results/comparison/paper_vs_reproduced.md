# Canonical Baseline Reproduction Comparison: Paper vs. Reproduced Results

**Research Paper**: *Social-Transmotion: Promptable Human Trajectory Prediction* (ICLR 2024)  
**Evaluation Protocol**: Official `evaluate_jta.py` and `evaluate_jrdb.py` (`--metric ade_fde --split test`)  
**Execution Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), Python 3.10.11, PyTorch 1.13.1+cu116, NumPy 1.26.4  
**Evaluation Date**: 2026-11-04  

---

## 1. Canonical Comparison Table

> **Precision and Reproduction Note**: The published ICLR 2024 paper reports evaluation metrics rounded to two decimal places (0.01 m). The underlying unrounded metrics from the original paper are not publicly available in the paper text or repository. Below, we report our exact unrounded floating-point evaluation results alongside the published rounded figures and the resulting raw metric deviations ($\Delta$). Because rounded FDE figures differ by 0.01 m in certain modalities (e.g., JTA Trajectory Only rounds to 1.99 m vs. published 1.98 m), these results represent a **close quantitative reproduction** rather than an exact decimal match. Deviations may reflect minor differences in hardware floating-point accumulation, dependency environment versions, or unrounded paper reporting thresholds.

| Dataset | Modality Setting | Modality Flag | Paper Reported (Rounded) | Reproduced ADE (Raw) | Reproduced FDE (Raw) | Raw $\Delta$ ADE | Raw $\Delta$ FDE | Reproduction Assessment |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **JTA** | Trajectory Only | `traj` | 0.99 / 1.98 | **0.9862** | **1.9891** | -0.0038 m | +0.0091 m | Close reproduction (raw rounds to 0.99 / 1.99) |
| **JTA** | T + 2D BB | `traj+2dbox` | 0.96 / 1.91 | **0.9571** | **1.9188** | -0.0029 m | +0.0088 m | Close reproduction (raw rounds to 0.96 / 1.92) |
| **JTA** | T + 2D Pose | `traj+2dpose` | 0.95 / 1.91 | **0.9511** | **1.9166** | +0.0011 m | +0.0066 m | Close reproduction (raw rounds to 0.95 / 1.92) |
| **JTA** | T + 3D Pose | `traj+3dpose` | 0.89 / 1.81 | **0.8874** | **1.8101** | -0.0026 m | +0.0001 m | Close reproduction (raw rounds to 0.89 / 1.81) |
| **JTA** | T + 3D Pose + 3D BB | `traj+3dpose+3dbox` | 0.89 / 1.81 | **0.8891** | **1.8144** | -0.0009 m | +0.0044 m | Close reproduction (raw rounds to 0.89 / 1.81) |
| **JTA** | All Modalities | `traj+all` | 0.88 / 1.80 | **0.8825** | **1.7957** | +0.0025 m | -0.0043 m | Close reproduction (raw rounds to 0.88 / 1.80) |
| **JRDB** | Trajectory Only | `traj` | 0.40 / 0.77 | **0.3972** | **0.7735** | -0.0028 m | +0.0035 m | Close reproduction (raw rounds to 0.40 / 0.77) |
| **JRDB** | T + 2D BB | `traj+2dbox` | 0.37 / 0.73 | **0.3712** | **0.7279** | +0.0012 m | -0.0021 m | Close reproduction (raw rounds to 0.37 / 0.73) |

---

## 2. Discrepancy & Tolerance Summary

- Across all eight evaluation configurations:
  - **Maximum Raw ADE Deviation**: $|\Delta \text{ADE}| \le 0.0038$ meters ($3.8$ mm).
  - **Maximum Raw FDE Deviation**: $|\Delta \text{FDE}| \le 0.0091$ meters ($9.1$ mm).
- **Scientific Takeaway**: All evaluated modalities closely replicate the trends, relative rankings, and absolute values reported in the original paper. Because underlying unrounded paper values are unavailable, we characterize these experiments as a faithful, close quantitative reproduction establishing an empirical baseline.
