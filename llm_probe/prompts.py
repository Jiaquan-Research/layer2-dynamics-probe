"""
Centralized prompt definitions for Layer-2 dynamical regime experiments.

This file intentionally contains only static textual stimuli.
No logic, no control flow, no parameterization.

All experimental regimes should import prompts from here
to guarantee reproducibility and semantic transparency.
"""

# ---------------------------------------------------------------------
# Healthy Baseline
# ---------------------------------------------------------------------

HEALTHY_PROMPT = (
    "Explain the concept of entropy in one sentence."
)

# ---------------------------------------------------------------------
# Lock-in Attractor (Recursive Collapse)
# ---------------------------------------------------------------------

LOCK_IN_PROMPT = (
    '{"level1": {"level2": {"level3": '
)

# ---------------------------------------------------------------------
# Decoupled Oscillation (Orthogonal Semantic Switching)
# ---------------------------------------------------------------------

DECOUPLED_JSON_PROMPT = (
    "Generate a JSON object for a user profile. Key: "
)

DECOUPLED_POEM_PROMPT = (
    "Write a dream-like poem about neon lights. Line: "
)

# ---------------------------------------------------------------------
# Geodesic Flow (Constructive Reasoning)
# ---------------------------------------------------------------------

GEODESIC_PROMPT = (
    "Starting from the basic axioms of Euclidean geometry, "
    "strictly deduce step-by-step why the sum of angles in a triangle is 180 degrees. "
    "Do not skip steps. Format the proof as a numbered list of logical implications.\n"
)
