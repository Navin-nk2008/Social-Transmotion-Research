# Original Architecture Analysis: Social-Transmotion (ICLR 2024)

## 1. Executive Summary & Overview

Social-Transmotion introduces a prompt-like transformer architecture for multi-modal human trajectory prediction. Rather than requiring specialized hand-crafted encoders for distinct visual cues (such as 2D/3D bounding boxes, 2D/3D human poses, and historical trajectories), the model unifies all visual observations into discrete modality tokens with identical hidden dimensionality $d = 128$.

The core processing pipeline consists of two cascaded transformer modules:
1. **Cross-Modality Transformer (CMT)** (`local_former`): Computes intra-person, inter-modality self-attention across temporal frames and visual cue tokens.
2. **Social Transformer (ST)** (`global_former`): Computes inter-person, social self-attention across pedestrians in the scene.

Below is an exhaustive, code-verified audit of the tensor transformations, dimensional mappings, and implementation mechanisms in `model_jta.py` and `model_jrdb.py`.

---

## 2. Input Data Representation and Token Organization

### 2.1 Coordinate Organization
Input sequences span $T_{obs} = 9$ observed samples ($t = 0 \dots 8$) and $T_{pred} = 12$ future prediction query samples ($t = 9 \dots 20$), yielding a total sequence window of $F = 21$ samples at 2.5 FPS (0.4 s sampling interval).

Batch input shape to model:
$$\text{tgt} \in \mathbb{R}^{B \times F_{in} \times (N \cdot J) \times K}$$
where:
- $B$: Batch size (e.g. 4 for JTA, 16 for JRDB in training; 1 or batch size during evaluation).
- $F_{in} = 9$ (observation horizon).
- $N$: Number of pedestrians in the scene.
- $J$: Number of tokens per pedestrian ($J = 47$ for JTA, $J = 2$ for JRDB).
- $K$: Coordinate channel dimension ($K = 4$).

### 2.2 Padding and Prediction Window Initialization
In `TransMotion.forward(tgt, padding_mask)`:
1. The future frames ($T_{pred} = 12$) are padded by replicating the last observed frame ($t = 8$):
   ```python
   pad_idx = np.repeat([in_F - 1], out_F) # 12 copies of index 8
   i_idx = np.append(np.arange(0, in_F), pad_idx) # [0, 1, ..., 8, 8, 8, ..., 8] (length 21)
   tgt = tgt[:, i_idx] # Shape: [B, 21, N*J, K]
   tgt = tgt.reshape(B, F, N, J, K) # Shape: [B, 21, N, J, K]
   ```
2. Ground-truth Pelvis Coordinate Centering:
   In `batch_process_coords`:
   - Primary trajectory coordinate centering: all coordinates are shifted relative to the primary pedestrian's pelvis at $t = T_{obs}-1$ ($t=8$).
   - Visual cues ($j \ge 1$) are shifted relative to each pedestrian's own position at $t=8$.

---

## 3. Modality Tokens and Feature Encoders

### 3.1 JTA Modality Layout ($J = 47$ tokens)
In JTA, each pedestrian has 47 tokens per frame:
- **Token 0**: Trajectory $(x, y)$ position ground coordinates ($K=2$ used, padded to 4).
- **Token 1**: 3D Bounding Box $(h, w, l, \text{rot}_z)$ ($K=4$).
- **Token 2**: 2D Bounding Box $(bb_{left}, bb_{top}, bb_{width}, bb_{height})$ ($K=4$).
- **Tokens 3–24 (22 tokens)**: 3D Body Pose joints $(x_j, y_j, z_j)$ for 22 body joints ($K=3$ used).
- **Tokens 25–46 (22 tokens)**: 2D Body Pose joints $(xx_j, yy_j)$ for 22 body joints ($K=2$ used).

Linear input projections:
- `self.fc_in_traj = nn.Linear(2, nhid)`: projects $(x, y) \to 128$.
- `self.fc_in_3dbb = nn.Linear(4, nhid)`: projects $(h, w, l, \text{rot}_z) \to 128$.
- `self.fc_in_2dbb = nn.Linear(4, nhid)`: projects 2D bbox $\to 128$.
- `self.fc_in_3dpose = nn.Linear(3, nhid)`: projects 3D joint $\to 128$.
- `self.fc_in_2dpose = nn.Linear(2, nhid)`: projects 2D joint $\to 128$.

### 3.2 JRDB Modality Layout ($J = 2$ tokens)
In JRDB, only 2D bounding boxes are available in addition to trajectory:
- **Token 0**: Trajectory $(x, y)$ ground coordinates ($K=2$).
- **Token 1**: 2D Bounding Box $(bb_{left}, bb_{top}, bb_{width}, bb_{height})$ ($K=4$), rescaled by $0.25$.
Linear projections:
- `self.fc_in_traj = nn.Linear(2, nhid)`: projects $(x, y) \to 128$.
- `self.fc_in_2dbb = nn.Linear(4, nhid)`: projects 2D bbox $\to 128$.

---

## 4. Positional, Identity, and Modality Encodings

The model incorporates distinct embedding layers for temporal sequence and identity:

### 4.1 Trajectory Temporal + Person ID Encoding (`LearnedTrajandIDEncoding`)
Applied to `tgt_traj` of shape $[B, F, N, d]$ (where $F=21$, $d=128$):
- Half dimension: $d/2 = 64$.
- `learned_encoding = nn.Embedding(21, 64)`: Temporal positional embedding for all 21 frames (both observation and future query frames).
- `person_encoding = nn.Embedding(1000, 64)`: Distinguishes pedestrian identity among up to 1000 concurrent agents.
- Interleaved injection:
  ```python
  x[:, :, :, 0:128:2] += learned_encoding(torch.arange(21)).unsqueeze(1).unsqueeze(0) # Even indices
  x[:, :, :, 1:128:2] += person_encoding(torch.arange(num_people)...).unsqueeze(0)    # Odd indices
  ```

### 4.2 Visual Cues Positional Encodings
Visual cues are observed only during the $T_{obs} = 9$ observation frames:
- 3D Bounding Box: `Learnedbb3dEncoding` (`nn.Embedding(9, 128)`).
- 2D Bounding Box: `Learnedbb2dEncoding` (`nn.Embedding(9, 128)`).
- 3D Pose: `Learnedpose3dEncoding` (`nn.Embedding(198, 128)`):
  Notice that $198 = 9 \text{ frames} \times 22 \text{ joints}$. Each joint across each time step receives a unique positional/identity embedding.
- 2D Pose: `Learnedpose2dEncoding` (`nn.Embedding(198, 128)`):
  Similarly $198 = 9 \times 22$ unique embeddings for 2D keypoints across time.

---

## 5. Sequence Flattening and Cross-Modality Transformer (CMT)

### 5.1 Local Sequence Concatenation and Token Arithmetic
The tokens are reshaped and concatenated along the sequence dimension:
- Trajectory tokens: $[F, B \cdot N, d] = [21, B \cdot N, 128]$.
- Visual cues (observed only for $T_{obs} = 9$):
  - In JTA:
    - 3D BBox: $[9, B \cdot N, 128]$
    - 2D BBox: $[9, B \cdot N, 128]$
    - 3D Pose: $[198, B \cdot N, 128]$ ($9 \times 22$)
    - 2D Pose: $[198, B \cdot N, 128]$ ($9 \times 22$)
    - **Modality Token Arithmetic**:
      - *Trajectory + 3D Pose Configuration*:
        $$L_{T+3DPose} = 21 \text{ (traj)} + 198 \text{ (3D pose)} = \mathbf{219 \text{ tokens}}$$
        *(Matches `seq_len: 219` documented in `configs/jta_all_visual_cues.yaml`).*
      - *All Modalities Configuration*:
        $$L_{All} = 21 + 9 + 9 + 198 + 198 = \mathbf{435 \text{ tokens per pedestrian}}$$
        *(Matches `MODEL.seq_len: 435` in `configs/jta_all_visual_cues.yaml`).*
      - *Token Delta and Relative Increase*:
        $$\Delta L = 435 - 219 = \mathbf{216 \text{ tokens}} \quad \left(\frac{216}{219} \approx \mathbf{+98.6\%}\right)$$
  - In JRDB:
    - 2D BBox: $[9, B \cdot N, 128]$
    - Concatenated total local sequence length:
      $$L_{local} = 21 + 9 = \mathbf{30 \text{ tokens per pedestrian}}$$
      *(Matches `MODEL.seq_len: 30` in `configs/jrdb_2dbox.yaml`).*

### 5.2 Cross-Modality Transformer Architecture
- Module: `self.local_former = AuxilliaryEncoderCMT(...)`
- Standard PyTorch `TransformerEncoderLayer`:
  - `d_model`: $128$
  - `nhead`: $4$ ($d_k = 32$ per head)
  - `dim_feedforward`: $1024$
  - `dropout`: $0.1$
  - `activation`: ReLU
- Layer count: `num_layers_local = 6`
- Self-attention operates across all $L_{local}$ tokens of each pedestrian simultaneously.
- Residual connection:
  $$\text{out}_{local} = \text{local\_former}(\text{tgt}) \cdot \text{output\_scale} + \text{tgt}$$
  where `output_scale = 1`.

---

## 6. Social Transformer (ST)

### 6.1 Extraction of Trajectory Queries
Only the trajectory tokens corresponding to the $F=21$ time steps are extracted from the local representation for social interaction modeling:
```python
out_local = out_local[:21] # Shape: [21, B*N, 128]
out_local = out_local.reshape(21, B, N, nhid).permute(2, 0, 1, 3).reshape(-1, B, nhid)
```
After permutation to $(N, 21, B, d)$ and reshaping:
$$\text{Input to ST} \in \mathbb{R}^{(N \cdot 21) \times B \times 128}$$
Here, the sequence length for the Social Transformer is $L_{global} = N \cdot 21$ (all pedestrians $\times$ all 21 frames).

### 6.2 Social Transformer Architecture
- Module: `self.global_former = AuxilliaryEncoderST(...)`
- Layer specifications:
  - `d_model`: $128$
  - `nhead`: $4$
  - `dim_feedforward`: $1024$
  - `dropout`: $0.1$
  - `activation`: ReLU
- Layer count: `num_layers_global = 3`
- Residual connection:
  $$\text{out}_{global} = \text{global\_former}(\text{out}_{local}) \cdot \text{output\_scale} + \text{out}_{local}$$

---

## 7. Prediction Head & Output Generation

### 7.1 Primary Pedestrian Trajectory Extraction
The target of prediction is the primary pedestrian ($n = 0$):
```python
out_primary = out_global.reshape(N, F, B, nhid)[0] # Shape: [21, B, 128]
out_primary = self.fc_out_traj(out_primary)        # Shape: [21, B, 2]
out = out_primary.transpose(0, 1).reshape(B, F, 1, 2) # Shape: [B, 21, 1, 2]
```
The final 12 frames ($t = 9 \dots 20$) are extracted as the predicted future 2D coordinates:
$$\hat{Y} \in \mathbb{R}^{B \times 12 \times 1 \times 2}$$

---

## 8. Systematic Comparison: JTA vs. JRDB Implementations

| Attribute | JTA Implementation (`model_jta.py`) | JRDB Implementation (`model_jrdb.py`) |
| :--- | :--- | :--- |
| **Input Modalities** | Trajectory, 3D BBox, 2D BBox, 3D Pose (22 joints), 2D Pose (22 joints) | Trajectory, 2D BBox |
| **Token Count ($J$)** | 47 | 2 |
| **Local Seq Length ($L_{local}$)** | 435 ($21 + 9 + 9 + 198 + 198$) | 30 ($21 + 9$) |
| **Pose Encoders** | `fc_in_3dpose` (3 $\to$ 128), `fc_in_2dpose` (2 $\to$ 128) | None |
| **3D BBox Encoder** | `fc_in_3dbb` (4 $\to$ 128) | None |
| **2D BBox Encoder** | `fc_in_2dbb` (4 $\to$ 128) | `fc_in_2dbb` (4 $\to$ 128) |
| **CMT Layers** | 6 | 6 |
| **ST Layers** | 3 | 3 |
| **Hidden Dim ($d$)** | 128 | 128 |
| **Feedforward Dim** | 1024 | 1024 |
| **Attention Heads** | 4 | 4 |
| **BBox Coordinate Rescaling** | None (direct coordinates) | Multiplied by $0.25$ |
| **Training Rotation Augmentation** | None in `batch_process_coords` | `getRandomRotatePoseTransform` applied |

---

## 9. Computational Bottleneck Verification

1. **Token Allocation vs. Attention Complexity in CMT**:
   - **Token Proportion**: In JTA (All Modalities), pose keypoint tokens occupy $396 / 435 = \mathbf{91.03\%}$ of the total sequence tokens per agent (22 keypoints $\times$ 9 frames $\times$ 2 modalities: 3D pose + 2D pose). Bounding box tokens occupy $18 / 435 = 4.14\%$, and trajectory tokens occupy $21 / 435 = 4.83\%$. Thus, 91.03% represents the proportion of sequence tokens occupied by pose features, not the proportion of total computational load.
   - **Attention Coefficients**: The CMT has $h = 4$ attention heads (`nhead: 4`) and $L_{local} = 435$.
     - Each head computes an attention coefficient matrix of shape $435 \times 435 = \mathbf{189,225}$ per-head attention coefficients.
     - Across all 4 attention heads, each layer computes $4 \times 189,225 = \mathbf{756,900}$ attention coefficients per pedestrian.
     - Across all 6 CMT layers, this sums to $6 \times 756,900 = \mathbf{4,541,400}$ attention coefficients per pedestrian (yielding $45,414,000$ coefficients across $N = 10$ pedestrians per batch element).
   - **Theoretical Attention Complexity vs. System Execution**: While the theoretical self-attention matrix size scales quadratically ($\mathcal{O}(L^2)$), transformer feedforward networks ($d \to d_{ff} \to d$) and projection layers scale linearly with token length ($\mathcal{O}(L)$). System-level FLOPs and wall-clock latency improvements from token reduction must be empirically profiled rather than assumed to scale strictly quadratically.

2. **Global Social Attention Complexity**:
   - ST sequence length is $L_{global} = N \cdot 21$ (e.g., for $N = 20$ pedestrians, $L_{global} = 420$ tokens).
   - Across $h = 4$ heads and 3 layers, the ST performs self-attention across all temporal queries of all agents concurrently. In dense crowds ($N \ge 20$), ST attention scaling becomes significant alongside CMT local computation.

