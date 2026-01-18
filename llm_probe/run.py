"""
Experiment runner for Layer-2 dynamical regimes.

This script orchestrates controlled generation scenarios
to elicit distinct dynamical behaviors in autoregressive LLMs.

It intentionally contains:
- NO prompt definitions
- NO control or intervention logic
- NO analysis or visualization code

Its sole responsibility is to produce reproducible logs.
"""

from llm_probe.wrapper import LLMWrapper
from llm_probe.probes import ProbeComputer
from llm_probe.prompts import (
    HEALTHY_PROMPT,
    LOCK_IN_PROMPT,
    DECOUPLED_JSON_PROMPT,
    DECOUPLED_POEM_PROMPT,
    GEODESIC_PROMPT,
)

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def run_healthy(llm):
    """
    Regime: Healthy Baseline.

    Each generation step is treated as an independent episode.
    Probe history is reset before every generation to establish
    a reference entropy baseline without temporal coupling.
    """
    print("\n=== MODE: HEALTHY (RESET) | Steps: 6 ===")

    probes = ProbeComputer()

    for step in range(6):
        probes.reset_history()
        out = llm.generate(
            HEALTHY_PROMPT,
            max_new_tokens=10,
            do_sample=False,
        )
        p = probes.compute(out["logits"], out["hidden"])

        print(
            f"[H-{step}] "
            f"Ent: {p['entropy']:.3f} | "
            f"Cons: None | "
            f"Out: {out['gen_text'][:40].strip()}..."
        )


def run_lock_in(llm):
    """
    Regime: Lock-in Attractor.

    A recursive prompt is used to induce collapse into a
    low-entropy, high-consistency attractor via autoregressive feedback.
    """
    print("\n=== MODE: LOCK-IN (RECURSIVE TRAP) | Steps: 15 ===")

    probes = ProbeComputer()
    probes.reset_history()

    curr = LOCK_IN_PROMPT

    for step in range(15):
        out = llm.generate(
            curr,
            max_new_tokens=8,
            do_sample=False,
        )
        p = probes.compute(out["logits"], out["hidden"])

        cons_str = (
            f"{p['consistency']:.4f}"
            if p["consistency"] is not None
            else "Init"
        )

        clean = out["gen_text"].strip().replace("\n", " ")

        print(
            f"[L-{step}] "
            f"Ent: {p['entropy']:.3f} | "
            f"Cons: {cons_str} | "
            f"Out: {clean}"
        )

        # Autoregressive feedback
        curr += out["gen_text"]


def run_decoupled_oscillation(llm):
    """
    Regime: Decoupled Oscillation.

    The model is forced to alternate between semantically
    orthogonal tasks at every step, preventing the formation
    of a stable latent trajectory.
    """
    print("\n=== MODE: DECOUPLED OSCILLATION | Steps: 20 ===")

    probes = ProbeComputer()
    probes.reset_history()

    curr = DECOUPLED_JSON_PROMPT

    for step in range(20):
        out = llm.generate(
            curr,
            max_new_tokens=16,
            do_sample=True,  # Sampling exaggerates oscillatory behavior
        )
        p = probes.compute(out["logits"], out["hidden"])

        cons_str = (
            f"{p['consistency']:.4f}"
            if p["consistency"] is not None
            else "Init"
        )

        clean = out["gen_text"].strip().replace("\n", " ")

        print(
            f"[D-{step}] "
            f"Ent: {p['entropy']:.3f} | "
            f"Cons: {cons_str} | "
            f"Out: {clean[:40]}..."
        )

        # Explicit semantic context switch
        curr = (
            DECOUPLED_POEM_PROMPT
            if step % 2 == 0
            else DECOUPLED_JSON_PROMPT
        )


def run_geodesic(llm):
    """
    Regime: Geodesic Flow (Constructive Reasoning).

    The prompt enforces long-horizon, step-by-step deduction.
    Expected signature:
    - Sustained high consistency (κ < 1)
    - Sawtooth entropy pattern (information injection → convergence)
    """
    print("\n=== MODE: GEODESIC (REASONING FLOW) | Steps: 20 ===")

    probes = ProbeComputer()
    probes.reset_history()

    curr = GEODESIC_PROMPT

    for step in range(20):
        out = llm.generate(
            curr,
            max_new_tokens=16,
            do_sample=False,  # Deterministic to isolate structure
        )
        p = probes.compute(out["logits"], out["hidden"])

        cons_str = (
            f"{p['consistency']:.4f}"
            if p["consistency"] is not None
            else "Init"
        )

        clean = out["gen_text"].strip().replace("\n", " ")

        print(
            f"[G-{step}] "
            f"Ent: {p['entropy']:.3f} | "
            f"Cons: {cons_str} | "
            f"Out: {clean[:40]}..."
        )

        curr += out["gen_text"]


def main():
    print(f"Target Model: {MODEL_ID}")
    llm = LLMWrapper(MODEL_ID)

    run_healthy(llm)
    run_lock_in(llm)
    run_decoupled_oscillation(llm)
    run_geodesic(llm)


if __name__ == "__main__":
    main()
