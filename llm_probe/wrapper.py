# llm_probe/wrapper.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMWrapper:
    """
    Thin inference wrapper for causal language models.

    Responsibilities:
    - Tokenization
    - Text generation (deterministic or stochastic)
    - Extraction of last-step logits and hidden states

    This wrapper intentionally contains NO control logic.
    It is designed to serve as a pure measurement interface
    for Layer-2 probes.
    """

    def __init__(self, model_id: str):
        print(f"Loading tokenizer: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        print(f"Loading model: {model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        do_sample: bool = False
    ):
        """
        Run a single generation step and return minimal
        internal signals required by Layer-2 probes.

        Args:
            prompt (str):
                Input prompt.
            max_new_tokens (int):
                Number of tokens to generate.
            do_sample (bool):
                If True, enable stochastic sampling.
                If False, use greedy decoding.

        Returns:
            dict with keys:
                - gen_text: generated text segment
                - logits: logits of the final generation step
                - hidden: hidden state of the final token
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        # Base generation configuration
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Conditional injection of sampling parameters
        if do_sample:
            gen_kwargs.update(dict(
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
            ))
        else:
            gen_kwargs.update(dict(
                do_sample=False,
            ))

        out = self.model.generate(
            **inputs,
            **gen_kwargs
        )

        # Decode only the newly generated tokens
        gen_sequences = out.sequences[:, inputs["input_ids"].shape[1]:]
        gen_text = self.tokenizer.decode(
            gen_sequences[0],
            skip_special_tokens=True
        )

        # Extract last-step signals for probes
        last_step_logits = out.scores[-1]
        last_step_hidden = out.hidden_states[-1][-1]

        return {
            "gen_text": gen_text,
            "logits": last_step_logits.unsqueeze(1),
            "hidden": last_step_hidden,
        }
