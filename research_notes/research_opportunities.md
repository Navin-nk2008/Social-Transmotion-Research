# Research Opportunities & Architecture Simplification Analysis
**Paper**: *Social-Transmotion: Promptable Human Trajectory Prediction* (ICLR 2024)  
**Authors**: Saeed Saadatnejad*, Yang Gao*, Kaouther Messaoud, Alexandre Alahi (VITA-EPFL)  
**Document**: `research_notes/research_opportunities.md`

---

## 1. Computational Profile of Social-Transmotion Baseline

Our codebase audit and experimental baseline evaluation reveal key computational characteristics:

### 1.1 Model Dimensions and Sequence Lengths
- **Hidden dimension**: $d = 128$
- **Feedforward dimension**: $d_{ff} = 1024$
- **Attention heads**: $h = 4$ ($d_k = 32$)
- **Local Transformer (CMT)**: $N_{layers} = 6$
- **Global Transformer (ST)**: $N_{layers} = 3$
- **Total Model Parameters**: $3,217,858$ (~3.22M parameters)

### 1.2 Sequence Length and Modality Token Arithmetic
In the JTA configuration:
- **Trajectory Only**: $L_{traj} = 21$ tokens (9 observed + 12 future queries).
- **Trajectory + 3D Pose**:
  $$L_{T+3DPose} = 21 \text{ (traj)} + (9 \times 22) \text{ (3D pose)} = 21 + 198 = \mathbf{219 \text{ tokens}}$$
  *(Matches `MODEL.seq_len: 219` specified in original config comments).*
- **All Modalities**:
  $$L_{All} = 21 \text{ (traj)} + 9 \text{ (3D BB)} + 9 \text{ (2D BB)} + 198 \text{ (3D pose)} + 198 \text{ (2D pose)} = \mathbf{435 \text{ tokens}}$$
- **Exact Token Delta**:
  $$\Delta L = 435 - 219 = \mathbf{216 \text{ tokens}}$$
- **Relative Increase**:
  $$\frac{216}{219} \approx \mathbf{+98.6\%}$$
  Stacking 2D pose, 2D bounding boxes, and 3D bounding boxes on top of 3D pose adds 216 tokens (+98.6% relative token increase) for a marginal change in reported accuracy (from 0.89 ADE down to 0.88 ADE, an improvement of just 0.01 m).

### 1.3 Theoretical Attention Complexity vs. System FLOPs
It is essential to distinguish between theoretical quadratic attention scaling and measured system FLOPs:
1. **Theoretical Attention Operation Scaling**:
   - The self-attention matrix size scales quadratically with sequence length: $\mathcal{O}(L^2 \cdot d)$.
   - For $L=435$, $L^2 = 189,225$ elements per head; for $L=219$, $L^2 = 47,961$ elements per head.
   - Reducing sequence length from $219 \to 75$ reduces the attention matrix by $(75/219)^2 \approx 11.7\%$, but this quadratic term represents only a portion of total layer computation.
2. **End-to-End FLOPs and Latency**:
   - Each transformer layer contains two feedforward projection layers ($d \to d_{ff} \to d$ with $d=128, d_{ff}=1024$), which scale linearly with $L$: $\mathcal{O}(2 \cdot L \cdot d \cdot d_{ff})$.
   - Feedforward MLPs account for $2 \times 128 \times 1024 = 262,144$ operations per token per layer, whereas attention projection and dot products scale as $4 \cdot L \cdot d^2 + 2 \cdot L^2 \cdot d$.
   - While token pruning reduces both attention operations $\mathcal{O}(L^2)$ and linear MLP operations $\mathcal{O}(L)$, overall end-to-end latency reductions are governed by memory bandwidth, kernel launch overheads, and hardware utilization. Latency speedups must therefore be measured empirically rather than inferred from theoretical complexity alone.

### 1.4 Redundancy in the Social Transformer (ST)
- ST sequence length:
  $$L_{global} = N \cdot F = 21 \cdot N$$
- For a dense scene with $N = 20$ agents:
  $$L_{global} = 420 \text{ tokens}$$
  Self-attention in ST computes interactions between pedestrian $A$ at frame $t_1$ and pedestrian $B$ at frame $t_2$. Because temporal dynamics within each pedestrian have already been modeled by the 6-layer CMT, full spatio-temporal social attention across all $(21 \cdot N)^2$ elements may introduce substantial redundancy.

---

## 2. Synthesis of Empirical Findings from the ICLR 2024 Paper

1. **Observed Pose Temporal Horizon (Table 4 in Original Paper)**:
   - Full 3D pose (9 observed frames): **ADE 0.89 m, FDE 1.81 m**.
   - Last observed pose only (frame $t=8$): **ADE 0.94 m, FDE 1.91 m**.
   - **Degradation**: Approximately **+5.6% ADE (+0.05 m)** and **+5.5% FDE (+0.10 m)**.
   - This comparison indicates that while the single last observed pose retains ~94.4% of full-horizon accuracy, the temporal history across the full 9 frames provides non-trivial predictive signal.
2. **Keypoint Attention Distribution**:
   - The paper's attention analysis (Section 4.3 and Appendix A.3) highlights that extremities (ankles, knees, wrists) receive prominent attention weights, whereas head-only pose achieves only 0.98 ADE / 1.98 FDE (virtually identical to trajectory-only 0.99 / 1.98).
   - This observation motivates investigating whether pruning to salient extremities can maintain predictive accuracy, which we formulate as an empirical research hypothesis.
3. **Marginal Returns of Visual Cue Stacking**:
   - Trajectory only: ADE 0.99 m, FDE 1.98 m.
   - Trajectory + 3D Pose (219 tokens): ADE 0.89 m, FDE 1.81 m (-10.1% ADE reduction).
   - Trajectory + All Visual Cues (435 tokens, +98.6% tokens): ADE 0.88 m, FDE 1.80 m (-1.1% incremental ADE reduction).
   - Stacking 216 additional visual tokens yielded only 0.01 m ADE reduction.
4. **Architecture Sequencing (CMT $\to$ ST)**:
   - CMT followed by ST is crucial; Table 5 shows reversing them (ST-CMT) yields 1.13 ADE (+22.8% error), and substituting CMT with an MLP (MLP-ST) yields 1.06 ADE (+15.2% error).
5. **Absence of Scene / Map Context**:
   - The authors explicitly identify the lack of scene geometry (walkways, obstacles) as a remaining challenge in multi-agent trajectory prediction.

---

## 3. Standardized Sparse-Pose Experimental Configurations

To systematically isolate the effects of skeletal joint pruning and temporal downsampling, we standardize on four explicit ablation configurations evaluated on the **JTA Dataset** in the **Trajectory + 3D Pose** setting:

1. **Configuration 1 (Full Baseline)**:
   - 22 joints $\times$ 9 observed frames ($L_{local} = 21 + 198 = \mathbf{219 \text{ tokens}}$).
   - Reference checkpoint metrics: ADE = 0.8874 m, FDE = 1.8101 m.
2. **Configuration 2 (Extremity Pruned, Full Horizon)**:
   - 6 selected joints $\times$ 9 observed frames ($L_{local} = 21 + 54 = \mathbf{75 \text{ tokens}}$).
   - Selected joints: Left/Right Ankle, Left/Right Wrist, Left/Right Knee (verified from the JTA skeleton coordinate layout: right wrist, left wrist, right knee, right ankle, left knee, left ankle).
3. **Configuration 3 (Full Skeleton, Last Observed Frame)**:
   - 22 joints $\times$ 1 final observed frame ($t = 8$) ($L_{local} = 21 + 22 = \mathbf{43 \text{ tokens}}$).
4. **Configuration 4 (Extremity Pruned, Last Observed Frame)**:
   - 6 selected joints $\times$ 1 final observed frame ($t = 8$) ($L_{local} = 21 + 6 = \mathbf{27 \text{ tokens}}$).

*Note on Additional Proposals*: Exploratory variants such as kinematic limb grouping, velocity delta tokens, or continuous latent bottlenecks are treated as separate follow-on investigations and are not conflated with this primary four-configuration benchmark.

---

## 4. Evaluation and Validation Protocol

- **Validation-Driven Selection**: Architecture comparison, hyperparameter validation, and initial ablation analysis will be performed strictly on the **JTA validation split (`val`, 10 NDJSON files, 3,565 scenes)**.
- **Locked Test Set**: The **JTA test split (12 NDJSON files, 5,021 scenes)** remains locked and will be evaluated once per configuration only after design parameters are fixed.
- **Hypothesis Framing**: Anticipated accuracy preservation and inference latency reductions are treated strictly as research hypotheses to be measured, not guaranteed bounds.
- **Novelty Assessment**: Potential contributions will be positioned relative to existing literature without unsupported claims of absolute precedence.
