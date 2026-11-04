# Reproduction Status Report: Social-Transmotion (ICLR 2024)

**Document**: `research_notes/reproduction_status.md`  
**Current Milestone**: **Original Checkpoint Evaluation and Computational Profiling Complete; Sparse-Pose Ablations Planned**  
**Lead Engineer**: Senior Research Engineer – Social-Transmotion Reproduction  
**Date**: 2026-11-04  
**Codebase**: `vita-epfl/social-transmotion`  

---

## 1. Project Phase Summary

1. **Repository Audit & Checkpoint Verification**: **Complete**. Both official checkpoints (`checkpoints_data/jta` and `checkpoints_data/jrdb`) verified, loaded on CPU, and confirmed with strict model parameter matching (3,217,858 parameters each).
2. **Dataset Configuration & Integrity**: **Complete**. Datasets configured via directory junctions (`data/jta_all_visual_cues` and `data/jrdb_2dbox`). 100% of all train, val, and test NDJSON files verified for syntax and complete feature coverage.
3. **Environment Setup**: **Complete**. Python 3.10.11 virtual environment configured with PyTorch 1.13.1+cu116 and pinned NumPy 1.26.4 for C-API compatibility.
4. **Original Checkpoint Evaluation**: **Complete**. All eight official evaluation configurations (six JTA and two JRDB) executed on `split: test` using official evaluation scripts. Results confirm a close quantitative reproduction of the published paper.
5. **Computational Complexity Profiling**: **Complete**. Forward pass latency, throughput (FPS), peak GPU memory, FLOPs, and MACs measured on the NVIDIA GeForce RTX 4060 Laptop GPU across $N \in [1, 2, 5, 10, 20]$ agents.
6. **Architecture Modification & Sparse-Pose Experiments**: **Planned / In Design**. An experimental specification for sparse-pose token pruning has been authored in `research_notes/experiments/sparse_pose_ablation_plan.md`. No new architecture has been implemented or experimentally validated yet.

---

## 2. Verified Pretrained Checkpoint Specifications

| Checkpoint Name | Checkpoint File Path | File Size | SHA-256 Checksum | Configured Architecture | Parameters | Key Status |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| **JTA Model** | `checkpoints_data/jta/checkpoint/checkpoint.pth.tar` | 37,712,843 bytes | `3F71704C...16CD73E7E` | `model_jta.py` (`jta_all_visual_cues`) | 3,217,858 | 0 missing, 0 unexpected |
| **JRDB Model**| `checkpoints_data/jrdb/checkpoint/checkpoint.pth.tar`| 37,280,495 bytes | `D7CD1733...EBFF8A204` | `model_jrdb.py` (`jrdb_2dbox`) | 3,217,858 | 0 missing, 0 unexpected |

---

## 3. Canonical Reproduction Results (Test Split)

> **Methodological Note**: Underlying unrounded paper metrics are unavailable. The values below compare published 2-decimal rounded figures with our unrounded raw evaluation metrics. Raw metric deviations remain small ($|\Delta \text{ADE}| \le 0.0038$ m, $|\Delta \text{FDE}| \le 0.0091$ m), demonstrating a close quantitative reproduction across all modalities.

| Dataset | Modality Setting | Modality Flag | Paper Reported (Rounded) | Reproduced (Raw) | Raw Deviation ($\Delta$) | Assessment |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **JTA** | Trajectory Only | `traj` | 0.99 / 1.98 | **0.9862 / 1.9891** | -0.0038 / +0.0091 | Close reproduction (raw rounds to 0.99 / 1.99) |
| **JTA** | T + 2D BB | `traj+2dbox` | 0.96 / 1.91 | **0.9571 / 1.9188** | -0.0029 / +0.0088 | Close reproduction (raw rounds to 0.96 / 1.92) |
| **JTA** | T + 2D Pose | `traj+2dpose` | 0.95 / 1.91 | **0.9511 / 1.9166** | +0.0011 / +0.0066 | Close reproduction (raw rounds to 0.95 / 1.92) |
| **JTA** | T + 3D Pose | `traj+3dpose` | 0.89 / 1.81 | **0.8874 / 1.8101** | -0.0026 / +0.0001 | Close reproduction (raw rounds to 0.89 / 1.81) |
| **JTA** | T + 3D P + 3D BB | `traj+3dpose+3dbox` | 0.89 / 1.81 | **0.8891 / 1.8144** | -0.0009 / +0.0044 | Close reproduction (raw rounds to 0.89 / 1.81) |
| **JTA** | All Modalities | `traj+all` | 0.88 / 1.80 | **0.8825 / 1.7957** | +0.0025 / -0.0043 | Close reproduction (raw rounds to 0.88 / 1.80) |
| **JRDB** | Trajectory Only | `traj` | 0.40 / 0.77 | **0.3972 / 0.7735** | -0.0028 / +0.0035 | Close reproduction (raw rounds to 0.40 / 0.77) |
| **JRDB** | T + 2D BB | `traj+2dbox` | 0.37 / 0.73 | **0.3712 / 0.7279** | +0.0012 / -0.0021 | Close reproduction (raw rounds to 0.37 / 0.73) |

---

## 4. Current Work State & Next Planned Actions

- **Current State**: The baseline reproduction and computational profiling phases are complete. All original source files, checkpoints, datasets, and logs remain untouched.
- **Next Phase**: Review the computational profile and experiment plan, implement the isolated experimental scripts in `LiteSocial/`, and commence the sparse-pose ablation experiments on the validation split.
