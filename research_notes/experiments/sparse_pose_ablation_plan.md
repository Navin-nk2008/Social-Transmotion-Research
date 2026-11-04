# Sparse-Pose Ablation Plan & Research Experiment Protocol

**Document**: `research_notes/experiments/sparse_pose_ablation_plan.md`  
**Purpose**: Rigorous experimental design and protocol for token-pruned sparse-pose trajectory forecasting.  
**Primary Reference Baseline**: JTA Trajectory + 3D Pose (`traj+3dpose`)  
**Frozen Baseline Checkpoint Metrics (Test Split)**:
- **ADE**: `0.8874` m (Paper reported: `0.89` m)
- **FDE**: `1.8101` m (Paper reported: `1.81` m)
- **Token Count**: 219 tokens ($L_{local} = 21 + 198$)

---

## 1. Physical Token-Pruning Principle

> **Fundamental Principle**: In transformer architectures, true computational efficiency requires **physically truncating sequence length $L$**. Merely zeroing coordinates or masking self-attention matrices while maintaining dummy tokens retains $\mathcal{O}(L^2)$ attention matrix allocation and $\mathcal{O}(L \cdot d \cdot d_{ff})$ feedforward layer FLOPs.  
> 
> In all experimental configurations below, the unused joints and temporal frames are **physically sliced out before linear projection**, yielding compact token sequences that reduce memory allocation, quadratic attention overhead, and linear MLP compute.

---

## 2. Four Structured Ablation Configurations

All ablations operate on the **JTA Dataset** in the **Trajectory + 3D Pose** setting.

```
                    ┌─────────────────────────────────────────────────────────┐
                    │               Skeletal Keypoint Scope                   │
                    ├──────────────────────────┬──────────────────────────────┤
                    │   Full 22 Joints         │   6 Salient Extremities      │
┌───────────────────┼──────────────────────────┼──────────────────────────────┤
│ 9 Observed Frames │ Configuration A (Full)   │ Configuration B (Extremities)│
│ (T_obs = 9)       │ L_local = 219 tokens     │ L_local = 75 tokens          │
├───────────────────┼──────────────────────────┼──────────────────────────────┤
│ Last Frame Only   │ Configuration C (Snap)   │ Configuration D (UltraLite)  │
│ (T_obs = 1, t=8)  │ L_local = 43 tokens      │ L_local = 27 tokens          │
└───────────────────┴──────────────────────────┴──────────────────────────────┘
```

### Detailed Configuration Breakdown

#### Configuration A: Full Baseline (Reference)
- **Joints**: All 22 skeletal joints.
- **Temporal Horizon**: 9 observed frames ($t = 0 \dots 8$).
- **Pose Tokens**: $22 \text{ joints} \times 9 \text{ frames} = 198$ tokens.
- **Trajectory Tokens**: 21 tokens (9 observed + 12 future query frames).
- **Total Local Sequence Length ($L_{local}$)**: $21 + 198 = \mathbf{219 \text{ tokens}}$.
- **Learned Positional Encoding**: `Learnedpose3dEncoding` initialized with `seq_len = 198`.
- **Primary Metric Target**: ADE $\le 0.8874$ m, FDE $\le 1.8101$ m.

#### Configuration B: Kinematic Extremity Pruning (Multi-Frame)
- **Joints**: 6 salient locomotion extremities:
  - Left Ankle, Right Ankle
  - Left Knee, Right Knee
  - Left Wrist, Right Wrist
- **Temporal Horizon**: 9 observed frames ($t = 0 \dots 8$).
- **Pose Tokens**: $6 \text{ joints} \times 9 \text{ frames} = 54$ tokens.
- **Total Local Sequence Length ($L_{local}$)**: $21 + 54 = \mathbf{75 \text{ tokens}}$.
- **Sequence Contraction**: $-144 \text{ tokens}$ (**-65.8% reduction in token count**).
- **Theoretical Attention Scaling**: $\left(\frac{75}{219}\right)^2 \approx 0.1174 \implies \mathbf{88.3\% \text{ reduction in CMT attention operations}}$.
- **Learned Positional Encoding**: `Learnedpose3dEncoding` resized to `seq_len = 54`.

#### Configuration C: Temporal Snapshot Pruning (Full Joints, Single Frame)
- **Joints**: All 22 skeletal joints.
- **Temporal Horizon**: 1 frame (most recent observed frame $t = T_{obs}-1 = 8$).
- **Pose Tokens**: $22 \text{ joints} \times 1 \text{ frame} = 22$ tokens.
- **Total Local Sequence Length ($L_{local}$)**: $21 + 22 = \mathbf{43 \text{ tokens}}$.
- **Sequence Contraction**: $-176 \text{ tokens}$ (**-80.4% reduction in token count**).
- **Theoretical Attention Scaling**: $\left(\frac{43}{219}\right)^2 \approx 0.0385 \implies \mathbf{96.1\% \text{ reduction in CMT attention operations}}$.
- **Learned Positional Encoding**: `Learnedpose3dEncoding` resized to `seq_len = 22`.

#### Configuration D: UltraLite Pose (Extremities + Temporal Snapshot)
- **Joints**: 6 salient locomotion extremities.
- **Temporal Horizon**: 1 frame (most recent observed frame $t = 8$).
- **Pose Tokens**: $6 \text{ joints} \times 1 \text{ frame} = 6$ tokens.
- **Total Local Sequence Length ($L_{local}$)**: $21 + 6 = \mathbf{27 \text{ tokens}}$.
- **Sequence Contraction**: $-192 \text{ tokens}$ (**-87.7% reduction in token count**).
- **Theoretical Attention Scaling**: $\left(\frac{27}{219}\right)^2 \approx 0.0152 \implies \mathbf{98.5\% \text{ reduction in CMT attention operations}}$.
- **Learned Positional Encoding**: `Learnedpose3dEncoding` resized to `seq_len = 6`.

---

## 3. Implementation and Architecture Adaptation Requirements

To guarantee scientific validity without altering original files, all modified architectures will reside in an isolated experimental package (e.g., `LiteSocial/`):

1. **Pre-Projection Feature Slicing**:
   - The tensor slicing logic must extract keypoints prior to `fc_in_3dpose`:
     ```python
     # Example for 6 selected joints:
     selected_joint_indices = [3, 4, 8, 9, 13, 14] # Exact indices corresponding to ankles, knees, wrists
     tgt_3dpose_sliced = tgt_3dpose[:, :, :, selected_joint_indices, :3]
     ```
2. **Positional Encoding Adaptation**:
   - `Learnedpose3dEncoding` must be parameterized dynamically:
     ```python
     class AdaptedLearnedPose3dEncoding(nn.Module):
         def __init__(self, d_model=128, num_pose_tokens=54, dropout=0.1, device='cuda:0'):
             super().__init__()
             self.learned_encoding = nn.Embedding(num_pose_tokens, d_model, max_norm=True).to(device)
     ```
3. **Local Key Padding Mask Alignment**:
   - In `model_jta.py`, line 265:
     ```python
     tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(self.seq_len, dim=1)
     ```
     `self.seq_len` must exactly match the new compact sequence length ($L_{local} \in \{219, 75, 43, 27\}$).
4. **Social Transformer Decoupling Invariance**:
   - The Social Transformer consumes only the 21 trajectory tokens extracted via `out_local[:21]`. Because the trajectory tokens remain at the front of `out_local`, the ST input dimensions ($21 \cdot N \times B \times 128$) remain fully compatible and intact across all four configurations.

---

## 4. Research Experiment Protocol

### 4.1 Training Specification
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, $\epsilon = 10^{-8}$)
- **Initial Learning Rate**: $\eta = 1 \times 10^{-4}$
- **Learning Rate Schedule**: Step decay by $0.1$ at $80\%$ of total epochs (`lr_drop: True`)
- **Loss Function**: Mean Squared Error on 2D Pelvis trajectory coordinates ($t = 9 \dots 20$):
  $$\mathcal{L} = \frac{1}{B} \sum_{b=1}^B \frac{1}{12} \sum_{t=9}^{20} \|\hat{Y}_{b, t} - Y_{b, t}\|_2 \times 100$$
- **Auxiliary Loss Weight**: $0.2$
- **Gradient Clipping**: `max_grad_norm: 1.0`
- **Batch Size**: $4$ scenes per batch
- **Total Epochs**: $50$ epochs

### 4.2 Split and Validation Policy
- **Primary Tuning Split**: **Validation split (`val`, 10 NDJSON files, 3,565 scenes)**.
  All hyperparameter explorations, learning rate tuning, and architecture iterations must be evaluated on the validation set.
- **Test-Set Freezing**:
  The **test split (12 NDJSON files, 5,021 scenes)** is evaluated strictly **once** per candidate architecture after all design choices are finalized, preventing data leakage and test-set overfitting.

### 4.3 Multi-Seed Experimental Policy
To distinguish true architectural effects from stochastic training noise:
- Every configuration will be trained across **3 independent random seeds**:
  $$\text{Seeds} = [0, 42, 2024]$$
- Reported metrics must include **Mean $\pm$ Standard Deviation**:
  $$\text{ADE} = \mu_{ADE} \pm \sigma_{ADE}, \quad \text{FDE} = \mu_{FDE} \pm \sigma_{FDE}$$

### 4.4 Hardware and Profiling Methodology
- All profiling conducted on the dedicated **NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)**.
- **Warm-up**: 50 iterations prior to recording.
- **Latency**: 100 timed iterations using hardware-synchronized CUDA Events:
  ```python
  torch.cuda.synchronize()
  start.record()
  _ = model(x, mask)
  end.record()
  torch.cuda.synchronize()
  ```
- **Compute Profiling**:
  - GFLOPs and GMACs measured via `torch.profiler`.
  - Peak GPU memory measured via `torch.cuda.max_memory_allocated()`.
  - Forward-pass latency and FPS recorded across $N \in [1, 2, 5, 10, 20]$ agents.

---

## 5. Structured Ablation Reporting Table

The final research deliverable for this ablation series will populate the following standardized benchmark table:

| Configuration ID | Architecture Description | Observed Joints | Temporal Frames | Token Count ($L$) | Validation ADE (m) | Validation FDE (m) | Test ADE (m) | Test FDE (m) | GFLOPs ($N=10$) | Latency ($N=10$, ms) | Peak VRAM (MB) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A** | Full Baseline | 22 | 9 | 219 | *TBD* | *TBD* | 0.8874 | 1.8101 | *Profiled* | *Profiled* | *Profiled* |
| **Config B** | Extremity Pruned | 6 | 9 | 75 | *TBD* | *TBD* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **Config C** | Temporal Snapshot | 22 | 1 | 43 | *TBD* | *TBD* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **Config D** | UltraLite Pose | 6 | 1 | 27 | *TBD* | *TBD* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

---

## 6. Pre-Implementation Safety Protocol

- Original model files (`model_jta.py`, `model_jrdb.py`), training scripts (`train_jta.py`, `train_jrdb.py`), evaluation scripts (`evaluate_jta.py`, `evaluate_jrdb.py`), and dataset loaders remain strictly unaltered.
- The experimental models will be placed in `LiteSocial/` with dedicated configuration files (`configs/experiments/sparse_pose_*.yaml`).
- No training runs will commence until this plan and profiling report have been formally reviewed.

