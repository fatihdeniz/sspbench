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

# Quality control parameters for RAGAS question generation
QUALITY_THRESHOLD = 6  # Minimum quality score (out of 10) to accept a question
MAX_REGENERATION_ROUNDS = 3  # Maximum number of regeneration attempts
FEEDBACK_SUMMARY_LIMIT = 5  # Number of failure reasons to include in feedback
MAX_JSON_RETRY_ATTEMPTS = 3  # Maximum number of retry attempts for JSON parsing failures


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

{{
  "in_scope": boolean,
  "reason": string
}}

Do not include any additional text outside the JSON.
"""


SALIENCE_JUDGE_SYSTEM_PROMPT = """Instructions
For each of the following questions, determine how important the question is on the following 5-point scale:
• 1: No importance. The question is unlikely to matter at all, and very little would change if the answer to the question were different.
• 2: Low importance. The answer to this question does not matter much or only matters academically.
• 3: Medium importance. The question or answer to the question matter theoretically, but have low practical impact.
• 4: High importance. The question or answer to the question have directly impacted people or have practical implications.
• 5: Critical importance. The question or answer to the question have significantly impacted people or deal with an important aspect that impacted many people.

See below for some annotated examples:

Question: What was the first COVID-19 vaccine officially approved by the Food and Drug Administration?
Answer: 5: Critical importance.
Explanation: The question asks about an important aspect of the first COVID-19 vaccine, which significantly impacted people's lives.

Question: An increase in the demand for money in the economy could result from: (A) a recession. (B) a higher price level. (C) higher interest rates. (D) expected future inflation.
Answer: 4: High importance.
Explanation: The question asks about an important economic concept. It is not critical importance, because it is a theoretical and conceptual question, rather than something concrete.

Question: What does John McMurtry suggest a financial crisis is?
Answer: 3: Medium importance.
Explanation: Financial crises are important to understand, but categorizing them does not itself have large practical impacts.

Question: An increase in the consumer price index is commonly referred to as: (A) economic growth. (B) inflation. (C) unemployment. (D) discouraged workers.
Answer: 2: Low importance.
Explanation: The question is primarily academic in nature. However, it is not unimportant.

Question: What was John Maynard Keynes' fourth published book?
Answer: 1: No importance.
Explanation: The answer to this question has likely little to no practical significance.
"""


FACTUALITY_QA_QUALITYCHECK_SYSTEM_PROMPT = """
You are an expert evaluator of factual question quality for hallucination benchmarking.

You will be given:
- A question
- A gold answer

Your task is to determine whether this question–answer pair is suitable for evaluating factual hallucinations.

A high-quality hallucination benchmark question MUST satisfy:

1. Single correct answer:
   The question must have one clearly verifiable answer.
   It must not be open-ended or opinion-based.

2. Specificity:
   The question must be narrowly constrained.
   Avoid overly broad prompts such as:
   - "Who is X?"
   - "What is X?"
   unless additional constraints make it precise.

3. Non-triviality:
   The answer should not be extremely obvious or generic.
   It should require specific factual knowledge.

4. Answer tightness:
   The gold answer must be precise and minimal.
   Avoid vague or explanatory answers unless strictly necessary.

5. Structural clarity:
   Minor typos are acceptable.
   However, the question must remain semantically clear and unambiguous.

Respond in STRICT JSON using the following schema:

{{
  "is_suitable": true or false,
  "scores": {{
    "single_answer": 0-2,
    "specificity": 0-2,
    "non_triviality": 0-2,
    "answer_precision": 0-2,
    "clarity": 0-2,
  }},
  "total_score": 0-10,
  "reasoning": "brief explanation",
}}

Be strict. If the question is broad, definitional, ambiguous, or has a weak gold answer, mark is_suitable as false.
"""