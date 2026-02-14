"""
Safety Novelty Engine - Adaptive Safety Benchmark Generation

This module implements the safety novelty engine for generating dynamic safety
benchmarks using iterative refinement.  It mirrors the factuality novelty engine
(sspbench.novelty) but targets safety-alignment evaluation instead of factual
hallucination detection.

Pipeline overview
-----------------
1. **Seed extraction** – parse existing safety datasets (aiXamine safety-alignment
   prompts) to extract harm-category seeds and existing prompts.
2. **Category generation / refinement** – use an LLM to produce or refine harm
   categories conditioned on target refusal-rate and prior iteration history.
3. **Prompt generation** – generate novel safety-test prompts per category,
   optionally grounding them in real-world news via web search.
4. **Scope & quality filtering** – LLM-as-judge checks whether generated prompts
   are (a) genuinely unsafe / boundary-testing and (b) high quality.
5. **Model evaluation** – test a target LLM, score its responses with a safety
   judge, and feed accuracy back into the next iteration.
"""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependency through llm_utils → novelty → llm_utils
# Users should import from sub-modules directly:
#   from sspbench.safety.safety_config import ...
#   from sspbench.safety.safety_core import ...
#   from sspbench.safety.safety_engine import ...
#   from sspbench.safety.safety_eval import ...
#   from sspbench.safety.safety_seeds import ...
