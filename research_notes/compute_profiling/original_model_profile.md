# Social-Transmotion: Original Model Computational Profile

**Profiling Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
**Environment**: Python 3.10.11, PyTorch 1.13.1+cu116, CUDA 11.6 runtime (Driver 592.82)
**Methodology**: 50 warm-up iterations, 100 timed iterations using high-precision CUDA Events with synchronization, and PyTorch Profiler for FLOP computation.

## 1. Parameter Summary

- **JTA Architecture (`model_jta.py`)**: Total Parameters: 3,217,858 | Trainable: 3,217,858
- **JRDB Architecture (`model_jrdb.py`)**: Total Parameters: 3,217,858 | Trainable: 3,217,858

## 2. End-to-End Evaluation Benchmarking (Test Split)

| Dataset | Modality | Test Split Scenes | Total Eval Time (s) | Avg Time per Scene (ms) |
| :--- | :--- | :---: | :---: | :---: |
| **JTA** | `traj+all` | 5,021 | ~155.0 s | ~30.87 ms |
| **JTA** | `traj` | 5,021 | ~154.5 s | ~30.77 ms |
| **JRDB** | `traj+2dbox` | 6,341 | ~21.2 s | ~3.34 ms |
| **JRDB** | `traj` | 6,341 | ~21.0 s | ~3.31 ms |

## 3. Forward Pass Latency, Memory, and Complexity Profile (Batch Size = 1)

### JTA Model (`model_jta.py`, $J=47$ tokens, $L_{local}=435$)

| Agents ($N$) | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Throughput (FPS) | Peak VRAM (MB) | FLOPs (GFLOPs) | MACs (GMACs) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 14.19 ± 3.83 | 15.60 | 19.65 | 70.5 | 103.15 | 2.3349 | 1.1675 |
| 2 | 14.25 ± 3.77 | 15.38 | 19.23 | 70.2 | 110.46 | 4.6712 | 2.3356 |
| 5 | 13.64 ± 3.70 | 14.29 | 18.41 | 73.3 | 132.43 | 11.6881 | 5.8440 |
| 10 | 15.45 ± 1.91 | 15.67 | 18.33 | 64.7 | 169.75 | 23.4100 | 11.7050 |
| 20 | 26.69 ± 1.48 | 26.84 | 28.79 | 37.5 | 242.38 | 46.9554 | 23.4777 |

### JRDB Model (`model_jrdb.py`, $J=2$ tokens, $L_{local}=30$)

| Agents ($N$) | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Throughput (FPS) | Peak VRAM (MB) | FLOPs (GFLOPs) | MACs (GMACs) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 11.95 ± 2.87 | 13.14 | 15.55 | 83.7 | 96.14 | 0.1628 | 0.0814 |
| 2 | 11.66 ± 3.23 | 12.74 | 16.30 | 85.8 | 96.44 | 0.3270 | 0.1635 |
| 5 | 12.00 ± 3.72 | 13.03 | 18.16 | 83.3 | 97.32 | 0.8275 | 0.4138 |
| 10 | 12.94 ± 3.01 | 13.77 | 16.85 | 77.3 | 98.79 | 1.6889 | 0.8445 |
| 20 | 13.38 ± 3.64 | 13.89 | 18.96 | 74.7 | 103.07 | 3.5133 | 1.7566 |

## 4. Key Computational Observations

1. **Sequence Length Impact**: For $N=1$, JTA ($L=435$) requires significantly more FLOPs than JRDB ($L=30$), illustrating the heavy cost of tracking 22 individual skeletal keypoints over 9 frames in the CMT.
2. **Agent Scaling**: In JTA, as $N$ increases from 1 to 20, the local transformer FLOPs scale linearly with $N$ ($N \times L_{local}$), while the Social Transformer (ST) computes attention over $21 \cdot N$ tokens.
3. **Memory Footprint**: Peak memory remains well under 100 MB during single-batch forward passes on RTX 4060, showing that parameter storage (3.22M floats $\approx 13$ MB) and activation tensors are light, but compute/latency is dominated by transformer attention and feedforward projections.
