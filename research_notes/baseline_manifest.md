# Social-Transmotion Baseline Manifest (Frozen)

**Document Identifier**: `research_notes/baseline_manifest.md`  
**Purpose**: Immutable specification of the reproduced baseline, environment, weights, scripts, datasets, and benchmark results prior to any architectural experimentation.  
**Creation Date**: 2026-11-04  
**Project**: ICLR 2024 Social-Transmotion Reproduction & Improvement  

---

## 1. Source Repository & Release Snapshot

- **Official Repository**: `https://github.com/vita-epfl/social-transmotion`
- **Release Archive**: `social-transmotion-ckpt_data (1).tar.gz`
- **Release Version**: Official ICLR 2024 Open-Source Release (snapshot timestamp: 2023-12-27)
- **Git State**: Static release directory tree; no Git history embedded in downloaded archive.

---

## 2. Cryptographic Checksums (SHA-256)

### 2.1 Model Checkpoints
| Artifact | Relative Path | File Size | SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| **JTA Official Checkpoint** | `checkpoints_data/jta/checkpoint/checkpoint.pth.tar` | 37,712,843 bytes | `3F71704C643425A556CD3F25B921419331866AC1D8EEEF43D94F29A16CD73E7E` |
| **JRDB Official Checkpoint**| `checkpoints_data/jrdb/checkpoint/checkpoint.pth.tar`| 37,280,495 bytes | `D7CD173357D51365C7EE54917BC6971FFB1E756B000FDE8F3B5395EEBFF8A204` |

### 2.2 Original Model & Training Code
| File Name | Path | SHA-256 Hash |
| :--- | :--- | :--- |
| `model_jta.py` | `model_jta.py` | `A8441D51A30AE9C4E046CDFED072F7A77A74FCE1C31A35D4A4EF3E929AC53198` |
| `model_jrdb.py` | `model_jrdb.py` | `87AB7809431D943F1B94A62D92323641E2474D5030CA6569EF11888E08CC78F4` |
| `evaluate_jta.py` | `evaluate_jta.py` | `944178D88E67DF74D363D66AE0F637176A0B6B681765D8C9CB78EBD7B44A32C3` |
| `evaluate_jrdb.py` | `evaluate_jrdb.py` | `DC59D30505FF99FBD61A384F45C30CBDC726F884AF16C01E5F184A7CF78F1F65` |
| `dataset_jta.py` | `dataset_jta.py` | `849D34B02C39F6AE435A0EB50AFBEE830DFAE50F529993BB9643B91B801FEAC9` |
| `dataset_jrdb.py` | `dataset_jrdb.py` | `17A21AFD31A2BA4852A41A3B168225A6AF3408EF35AEFAD8715C2F3E0C2E3119` |
| `train_jta.py` | `train_jta.py` | `75C3919C2E5A1415F63335EBD91997819998184A8241539ABE8CFC3BAC6868C0` |
| `train_jrdb.py` | `train_jrdb.py` | `CCFEC30BB0DE7354DC347F51752B7137911BE48D4CCDD7DFB269C285D77A1EC8` |
| `configs/jta_all_visual_cues.yaml` | `configs/jta_all_visual_cues.yaml` | `446BE932D77B4F327732F88DBDC83A503DD675A5622AD063CF6C26A3C41A4301` |
| `configs/jrdb_2dbox.yaml` | `configs/jrdb_2dbox.yaml` | `63B928EAEB32716C3C07AE717410D0C453D506CA216EFA974C1E25E338FB9097` |

---

## 3. Verified Datasets & Split Locations

All preprocessed datasets are configured via NTFS directory junctions pointing directly to `checkpoints_data/`:
- `data/jta_all_visual_cues` $\iff$ `checkpoints_data/jta/data`
- `data/jrdb_2dbox` $\iff$ `checkpoints_data/jrdb/data`

| Dataset | Split | Physical Path | NDJSON Files | Scenes | Track Frames |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **JTA** | `train` | `data/jta_all_visual_cues/train` | 206 | 88,719 | 544,552 |
| **JTA** | `val` | `data/jta_all_visual_cues/val` | 10 | 3,565 | 22,544 |
| **JTA** | `test` | `data/jta_all_visual_cues/test` | 12 | 5,021 | 34,078 |
| **JRDB**| `train` | `data/jrdb_2dbox/train` | 10 | 19,843 | 54,880 |
| **JRDB**| `val` | `data/jrdb_2dbox/val` | 1 | 496 | 1,734 |
| **JRDB**| `test` | `data/jrdb_2dbox/test` | 2 | 6,341 | 8,943 |

---

## 4. Hardware and Runtime Environment

- **Operating System**: Microsoft Windows 11 Enterprise/Pro (build 10.0.26100, x86_64)
- **Python**: Python 3.10.11 (64-bit)
- **PyTorch**: 1.13.1+cu116
- **TorchVision**: 0.14.1+cu116
- **NumPy**: 1.26.4 (constrained `<2.0.0` for C-API stability)
- **CUDA Driver Version**: 592.82 (Driver CUDA: 13.1, Runtime: 11.6)
- **GPU Accelerator**: NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB GDDR6 VRAM)
- **Virtual Environment**: `.venv` located at `D:\Coding\Github\Social-Transmotion-Research\.venv`

---

## 5. Frozen Reproduction Benchmark

### 5.1 JTA Benchmark (Split: `test`, Horizon: 9 obs $\to$ 12 pred frames)
| Modality Evaluated | Modality Flag | Paper (Rounded) | Reproduced (Raw) | Raw Deviation ($\Delta$) | Exact Evaluation Command |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Trajectory Only** | `traj` | 0.99 / 1.98 | **0.9862 / 1.9891** | -0.0038 / +0.0091 | `.\.venv\Scripts\python.exe evaluate_jta.py --ckpt checkpoints_data/jta/checkpoint/checkpoint.pth.tar --metric ade_fde --split test --modality traj` |
| **T + 2D BB** | `traj+2dbox` | 0.96 / 1.91 | **0.9571 / 1.9188** | -0.0029 / +0.0088 | `.\.venv\Scripts\python.exe evaluate_jta.py --ckpt checkpoints_data/jta/checkpoint/checkpoint.pth.tar --metric ade_fde --split test --modality traj+2dbox` |
| **T + 2D Pose** | `traj+2dpose` | 0.95 / 1.91 | **0.9511 / 1.9166** | +0.0011 / +0.0066 | `.\.venv\Scripts\python.exe evaluate_jta.py --ckpt checkpoints_data/jta/checkpoint/checkpoint.pth.tar --metric ade_fde --split test --modality traj+2dpose` |
| **T + 3D Pose** | `traj+3dpose` | 0.89 / 1.81 | **0.8874 / 1.8101** | -0.0026 / +0.0001 | `.\.venv\Scripts\python.exe evaluate_jta.py --ckpt checkpoints_data/jta/checkpoint/checkpoint.pth.tar --metric ade_fde --split test --modality traj+3dpose` |
| **T + 3D P + 3D BB**| `traj+3dpose+3dbox` | 0.89 / 1.81 | **0.8891 / 1.8144** | -0.0009 / +0.0044 | `.\.venv\Scripts\python.exe evaluate_jta.py --ckpt checkpoints_data/jta/checkpoint/checkpoint.pth.tar --metric ade_fde --split test --modality traj+3dpose+3dbox` |
| **All Modalities** | `traj+all` | 0.88 / 1.80 | **0.8825 / 1.7957** | +0.0025 / -0.0043 | `.\.venv\Scripts\python.exe evaluate_jta.py --ckpt checkpoints_data/jta/checkpoint/checkpoint.pth.tar --metric ade_fde --split test --modality traj+all` |

### 5.2 JRDB Benchmark (Split: `test`, Horizon: 9 obs $\to$ 12 pred frames)
| Modality Evaluated | Modality Flag | Paper (Rounded) | Reproduced (Raw) | Raw Deviation ($\Delta$) | Exact Evaluation Command |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Trajectory Only** | `traj` | 0.40 / 0.77 | **0.3972 / 0.7735** | -0.0028 / +0.0035 | `.\.venv\Scripts\python.exe evaluate_jrdb.py --ckpt checkpoints_data/jrdb/checkpoint/checkpoint.pth.tar --metric ade_fde --split test --modality traj` |
| **T + 2D BB** | `traj+2dbox` | 0.37 / 0.73 | **0.3712 / 0.7279** | +0.0012 / -0.0021 | `.\.venv\Scripts\python.exe evaluate_jrdb.py --ckpt checkpoints_data/jrdb/checkpoint/checkpoint.pth.tar --metric ade_fde --split test --modality traj+2dbox` |

---

## 6. Baseline Freeze Declaration

This manifest establishes the verified, frozen baseline against which all subsequent lightweight and simplified architecture experiments will be evaluated.
No modifications will be made to original checkpoints, models, datasets, or evaluation scripts.

