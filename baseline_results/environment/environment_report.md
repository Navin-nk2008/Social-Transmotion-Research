# Environment Verification Report

**Date & Time**: 2026-11-04 09:17 UTC+5:30  
**Project**: Social-Transmotion (ICLR 2024 Reproduction)  
**Workspace**: `D:\Coding\Github\Social-Transmotion-Research`

---

## 1. System Hardware & Operating System

- **Operating System**: Microsoft Windows 11 (build 10.0.26100)
- **Host Architecture**: x86_64 / win_amd64
- **CPU**: Multi-core x86_64
- **System Memory**: 16 GB+ Host RAM

---

## 2. NVIDIA GPU & CUDA Driver

- **GPU Model**: NVIDIA GeForce RTX 4060 Laptop GPU
- **VRAM Total**: 8,188 MiB (8.0 GB GDDR6)
- **Driver Version**: 592.82
- **Driver CUDA Capability**: CUDA 13.1
- **PyTorch CUDA Build**: CUDA 11.6 (`cu116`)
- **CUDA Device Count**: 1
- **Device Identifier**: `cuda:0`
- **Functional CUDA Sanity Check**: Verified with tensor creation and matrix multiplication on `cuda:0` without errors.

---

## 3. Python Runtime & Virtual Environment

- **Python Version**: Python 3.10.11 (`tags/v3.10.11:7d4cc5a, Apr 5 2023, 00:38:17 [MSC v.1929 64 bit (AMD64)]`)
- **Virtual Environment**: `.venv` isolated workspace environment located at `D:\Coding\Github\Social-Transmotion-Research\.venv`
- **Interpreter Path**: `D:\Coding\Github\Social-Transmotion-Research\.venv\Scripts\python.exe`

---

## 4. Installed Packages & Compatibility Audit

| Package | Installed Version | Source / Note | Compatibility Status |
| :--- | :--- | :--- | :--- |
| `torch` | `1.13.1+cu116` | PyTorch official cu116 wheel | Compatible (matches official `requirements.txt`) |
| `torchvision` | `0.14.1+cu116` | PyTorch official cu116 wheel | Compatible |
| `torchaudio` | `0.13.1+cu116` | PyTorch official cu116 wheel | Compatible |
| `numpy` | `1.26.4` | PyPI | Critical: constrained to `<2.0.0` to preserve PyTorch 1.13 C-API `_ARRAY_API` tensor interop |
| `scipy` | `1.15.3` | PyPI | Compatible |
| `pyyaml` | `6.0.3` | PyPI | Compatible |
| `progress` | `1.6.1` | PyPI | Compatible |
| `pillow` | `12.3.0` | PyPI | Compatible |
| `tensorboard`| `2.21.0` | PyPI | Compatible |
| `pykalman` | `0.11.2` | PyPI | Compatible |
| `six` | `1.17.0` | PyPI | Compatible |

### Critical Dependency Resolution Note:
When `requirements.txt` was initially installed without a NumPy upper bound, pip resolved to `numpy==2.2.6` (released mid-2024). Because PyTorch 1.13.1 was built against NumPy 1.x, NumPy 2.x triggers `_ARRAY_API not found` and crashes `torch.from_numpy()` and `.numpy()`. Pinning `numpy<2` (`numpy==1.26.4`) completely resolved the issue without modifying any repository source code.

