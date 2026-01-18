# Law 3.1 — Deep Canals  
### Phase-Space Structure of Semantic Collapse

## Status
Empirical Law  
Validated via inference-time Layer-2 diagnostics

---

## 1. Statement of the Law

**Law 3.1 (Deep Canals):**

> In autoregressive large language models, semantic collapse manifests
> as attractor structures in a low-dimensional phase space defined by
> temporal structural continuity (κ) and predictive uncertainty (H).

Formally, during inference, the state trajectory  
Ψₜ = (κₜ, Hₜ)  
converges toward narrow basins (“canals”) characterized by:

- κ → 1 (near-perfect temporal self-alignment)  
- H → 0 (collapse of predictive uncertainty)

Once the trajectory enters such a basin, it exhibits **strong resistance
to semantic escape under purely autoregressive decoding dynamics**.

---

## 2. Motivation

Failure modes in large language models are commonly described using
surface-level symptoms such as:

- repetition  
- hallucination  
- overconfidence  
- infinite or circular loops  

While descriptive, these labels do not identify a **dynamical invariant**
shared across tasks or prompts.

Law 3.1 reframes semantic collapse as a **state-space phenomenon**:
a loss of effective degrees of freedom driven by self-reinforcing
temporal alignment during autoregressive inference.

---

## 3. Measurement Framework

Law 3.1 is evaluated using Layer-2 inference-time probes.

### 3.1 Entropy (H)

Shannon entropy of the next-token predictive distribution.

- High H → high local uncertainty and branching freedom  
- Low H → deterministic or near-deterministic continuation  

### 3.2 Consistency (κ)

Cosine similarity between consecutive final-layer hidden states.

- κ ≈ 0 → orthogonal or weakly coupled semantic steps  
- κ → 1 → strong temporal self-alignment of latent structure  

Together, (κ, H) define a minimal phase-space sufficient
to distinguish collapse from non-collapse dynamics.

---

## 4. Empirical Signature

A **Deep Canal** is identified by the following reproducible signature:

| Metric     | Observed Behavior                              |
|-----------|-----------------------------------------------|
| κ         | Monotonic increase toward saturation           |
| H         | Rapid decay toward near-zero values            |
| Trajectory| Convergence into a narrow phase-space basin    |

This pattern is consistently observed across prompts
that induce recursive, self-referential, or excessively
self-consistent generation.

---

## 5. Interpretation

Deep Canals are not caused by incorrect facts,
dataset sparsity, or lack of knowledge.

They arise from **structural dynamics intrinsic to
autoregressive decoding**, where high temporal alignment
progressively suppresses alternative semantic trajectories.

Once κ saturates, the system ceases to explore viable
semantic branches, even when those branches remain
statistically non-zero.

---

## 6. Scope and Limitations

- Law 3.1 describes **inference-time dynamics only**  
- No claims are made about training-time optimization or model weights  
- No mitigation, intervention, or control strategy is implied  

Resistance to escape is defined **relative to purely autoregressive
generation without external intervention**.

Law 3.1 is a diagnostic law, not a corrective one.

---

## 7. Relation to Other Regimes

Law 3.1 distinguishes collapse dynamics from other
Layer-2 regimes, including:

- **Healthy Baseline** — weak temporal coupling  
- **Decoupled Oscillation** — low κ with fluctuating H  
- **Geodesic Flow** — sustained high κ without saturation  

These regimes are defined formally in `terminology.md`.

---

## 8. Reproducibility

All observations supporting Law 3.1 are reproducible using:

```bash
python -m llm_probe.run
