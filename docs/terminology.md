# Terminology

## Entropy (H)
Shannon entropy of the next-token predictive distribution.
Measures local uncertainty.

## Consistency (κ)
Cosine similarity between consecutive final-layer hidden states.
Measures structural continuity of the latent trajectory.

## Regimes
- **Healthy Baseline**: Independent generations, no temporal coupling.
- **Lock-in Attractor**: High κ, low H. Collapse into repetitive structure.
- **Decoupled Oscillation**: Low κ, fluctuating H. Orthogonal task switching.
- **Geodesic Flow**: Sustained high κ (< 1), sawtooth entropy pattern.
