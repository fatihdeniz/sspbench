"""
Configuration and constants for the Novelty Engine.
"""

import os
from types import SimpleNamespace

# Default system message for JSON-based tasks
DEFAULT_JSON_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your reasoning and language skills.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
Reply "TERMINATE" in the end when everything is done.
"""

# Wikipedia API credentials (these should be moved to environment variables in production)
WIKIPEDIA_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIwMDFkMTFmNmQ2MzVmMGY4YmI3MDlkNWViN2ZhNDRlYiIsImp0aSI6IjMzOGQ0Mzc0YzNmZjE5NjBlZDkzNjIwNTdiYjMwYjExOWYzZTY2MzVkZjM3NmY3NDcyZjczMDcyMjNiYzU4ODFjODBkOTliOTZmMjAzZGNkIiwiaWF0IjoxNzEyNjEwMTg0LjY4OTIyNywibmJmIjoxNzEyNjEwMTg0LjY4OTIzLCJleHAiOjMzMjY5NTE4OTg0LjY4NzY1Mywic3ViIjoiNzUzODczODIiLCJpc3MiOiJodHRwczovL21ldGEud2lraW1lZGlhLm9yZyIsInJhdGVsaW1pdCI6eyJyZXF1ZXN0c19wZXJfdW5pdCI6NTAwMCwidW5pdCI6IkhPVVIifSwic2NvcGVzIjpbImJhc2ljIl19.YN0ZvSzsBuYe3Mg-r0C63cWxDXPU3GOCyspUqg4mMv27Qw1FJq9F9H6JKJAUMrqQxB-xyWZqpu8mekvMoxb3Ha5S2fpPbuM4gMB0JketqG2obaDd4QqgtJjg8KDYKwR8ieKoPRLDSHv3Tv4NcvIL-EvzjkRybqrukzQwttwuBUwxmlY8vhC1BZed7URt_-KhMYPsnNfJLSBeWivYJOmrqF2S04AOS0Egjul8Pz_yXAQ7q7aqpIwg6X2jod0ZN5h1gnmAvZmoLB7mKSAxrHEUL2zaQ8BVERWostWVA9ek556cuUJe5NusQ0XW7pcsYIi0YpFjKOBuq-tXzuOlbxFhlbwrp6xkhE_grQGNs1IxyT-w_sjQc2gI48FDe0ldDrTg6ZmgLELsjJM8xOxBy1ng1fY73p-QnaDdxX4hqRw2ZBDlZ1E2j84lvVrv62x_SHPiBNAeywEPcOqDRV_XbU6ArOyJ7QTZXRu9UOT0XDQ-Fx3maCRGb35W4aOtLSWL-SSXYLI8ZuOQ2BwKQQYYbEDMp0W7NjHWzh8YPv6Y2wDaMzsAqaxk2c36pNvTToiTc_P6_a56lydQwoT8ACx1kzzw5lTNPKPEPxPGNiMgtsL3VqtxJWMR7Lgq-ZKwI7cwQ5FTp2YriQDBYuvoaDQeG_eVh8BlNlyg26OYojtYbNos3os"
WIKIPEDIA_CLIENT_ID = "001d11f6d635f0f8bb709d5eb7fa44eb"

# CUDA device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "7")

# Sampling parameters
DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0,
    "repetition_penalty": 1.1,
    "max_tokens": 1024
}

def get_default_sampling_params():
    """Get default sampling parameters for LLM generation."""
    from vllm import SamplingParams
    return SamplingParams(**DEFAULT_SAMPLING_PARAMS)


FACTUALITY_QUESTION_GENERATOR_PROMPT = """
You are generating questions for a factuality hallucination evaluation.

Generate a single question that:
- Is fact-seeking and grounded in real-world knowledge
- Has a single, short, objective answer
- Tests factual recall only (no opinions, hypotheticals, creativity, or multi-step reasoning)
- Belongs to any real-world knowledge domain (e.g., science, geography, history, arts, politics, media, sports, technology)

The question must be clear, unambiguous, and answerable with a short factual response.
"""

FACTUALITY_QA_SCOPE_JUDGE_PROMPT = """
You are judging whether a question–answer pair is within the scope of a factuality hallucination test.

Question:
{question}

Answer:
{answer}

A pair is IN SCOPE if:
- The question is fact-seeking and refers to real-world knowledge
- The expected answer is a short, objective fact
- The question does NOT require opinions, speculation, hypotheticals, creativity, or multi-step reasoning
- The question fits within any real-world knowledge domain

Return your decision in the following STRICT JSON format only:

{
  "in_scope": boolean,
  "reason": string
}

Do not include any additional text outside the JSON.
"""
