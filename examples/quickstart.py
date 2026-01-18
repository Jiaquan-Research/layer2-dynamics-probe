from llm_probe.wrapper import LLMWrapper
from llm_probe.probes import ProbeComputer
from llm_probe.prompts import HEALTHY_PROMPT

llm = LLMWrapper("meta-llama/Meta-Llama-3-8B-Instruct")
probe = ProbeComputer()

out = llm.generate(HEALTHY_PROMPT, max_new_tokens=10, do_sample=False)
metrics = probe.compute(out["logits"], out["hidden"])

print("Generated:", out["gen_text"])
print("Entropy:", metrics["entropy"])
print("Consistency:", metrics["consistency"])
