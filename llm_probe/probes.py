import torch
import torch.nn.functional as F

class ProbeComputer:
    """
    A minimal, stateful probe for Layer-2 semantic dynamics.

    Responsibilities:
    1. Compute predictive uncertainty (Shannon Entropy) of the next-token distribution.
    2. Compute structural continuity (Cosine Consistency) of the latent trajectory.

    This class maintains an internal state (prev_embedding) to measure
    temporal coherence across generation steps.
    """

    def __init__(self):
        # Stores the embedding of the previous step
        # Used to compute temporal consistency (κ)
        self.prev_embedding = None

    def reset_history(self):
        """
        Explicitly reset the internal state.
        This defines an episode boundary and prevents
        cross-trajectory leakage between experiments.
        """
        self.prev_embedding = None

    def compute(self, logits, hidden):
        """
        Compute Layer-2 diagnostic metrics.

        Args:
            logits: Tensor of shape [1, T, vocab_size]
                    Raw model logits for the current generation step.
            hidden: Tensor of shape [1, T, hidden_dim]
                    Hidden states corresponding to the same step.

        Returns:
            dict with:
                - entropy (float): token-level predictive entropy.
                - consistency (float or None): cosine similarity with the previous step.
        """

        # ---- Entropy Probe (Uncertainty of next-token decision) ----
        # Force FP32 to avoid numerical instability in softmax.
        # Half precision (FP16/BF16) may cause overflow/underflow and NaNs.
        last_logits = logits[:, -1, :].float()  # [1, vocab_size]

        # Softmax in FP32 for numerical stability
        probs = F.softmax(last_logits, dim=-1)

        # Shannon entropy: H(x) = -sum(p * log(p))
        # Small epsilon (1e-12) added to avoid log(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()

        # ---- Consistency κ (Temporal structural continuity) ----
        # Extract the embedding of the last generated token
        emb = hidden[:, -1, :].float()  # [1, hidden_dim]

        consistency = None
        if self.prev_embedding is not None:
            # Cosine similarity between consecutive latent states
            # Measures the angle of semantic drift
            consistency = F.cosine_similarity(
                emb, self.prev_embedding, dim=-1
            ).item()

        # Update internal state (detach to avoid computation graph growth)
        self.prev_embedding = emb.detach()

        return {
            "entropy": entropy,
            "consistency": consistency,
        }