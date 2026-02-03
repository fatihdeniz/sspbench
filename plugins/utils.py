import os
import re
import gc
import html
import json
import subprocess
import contextlib
from typing import Any, List, Sequence, Tuple
from pandas.api.types import is_numeric_dtype

import ray
import torch
from openai import OpenAI
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

from constants import CodeSecurityTests

ATTACK_NAME = {
    "SEMEMEPSO": "SememePSO",
    "TEXTFOOLER": "TextFooler",
    "TEXTBUGGER": "TextBugger",
    "SEMATTACK": "SemAttack",
    "BERTATTACK": "BERTATTACK",
    "SCPN": "SCPN",
    "T3": "T3",
    "STRESSTEST": "StressTest",
    "CHECKLIST": "CheckList",
    "ADVSQUAD": "AdvSQuAD",
    "ADVFEVER": "AdvFever" 
}
NEGATIVE_PATTERN = re.compile(r"\b(?:negative|egative)\b", re.IGNORECASE)
POSITIVE_PATTERN = re.compile(r"\b(?:positive|positivo)\b", re.IGNORECASE)
NO_PATTERN = re.compile(r"\b(?:no|not|doesn't answer)\b", re.IGNORECASE)
MAYBE_PATTERN = re.compile(r"\bmaybe\b", re.IGNORECASE)
YES_PATTERN = re.compile(r"\b(?:yes|answers|does answer)\b", re.IGNORECASE)

def camel_case(string):
    words = re.split("_+", string)
    cc_string = words[0] + "".join(map(lambda word: word.title(), words[1:]))

    return cc_string

def parse_line_sentiment(line):
    positive = POSITIVE_PATTERN.search(line)
    negative = NEGATIVE_PATTERN.search(line)

    if positive and negative:
        return -1
    elif positive:
        return 1
    elif negative:
        return 0
    else:
        return -1

def parse_sentiment(response):
    clean_response = response.strip().lower()
    response_lines = clean_response.splitlines()
    if len(response_lines) > 0:
        first_line = response_lines[0].strip()
        final_line = response_lines[-1].strip()
    else:
        return -1
    
    last_sentiment = parse_line_sentiment(final_line)
    if last_sentiment == -1:
        first_sentiment = parse_line_sentiment(first_line)
        if first_sentiment == -1:
            full_sentiment = parse_line_sentiment(clean_response)
            return full_sentiment
        else:
            return first_sentiment
    else:
        return last_sentiment
    
def parse_line_answer(line, task_labels, maybe):
    if maybe:
        no = NO_PATTERN.search(line)
        maybe = MAYBE_PATTERN.search(line)
        yes = YES_PATTERN.search(line)
        if yes and not (maybe or no):
            return task_labels["yes"]
        elif no and not (yes or maybe):
            return task_labels["no"]
        elif maybe:
            return task_labels["maybe"]
        else:
            return -1
    else:
        no = NO_PATTERN.search(line)
        yes = YES_PATTERN.search(line)
        if yes and not no:
            return task_labels["yes"]
        elif not yes and no:
            return task_labels["no"]
        else:
            return -1
        
def parse_answer(response, task_labels, maybe):
    clean_response = response.strip().lower()
    response_lines = clean_response.splitlines()
    if len(response_lines) > 0:
        first_line = response_lines[0].strip()
        final_line = response_lines[-1].strip()
    else:
        return -1
    
    last_label = parse_line_answer(final_line, task_labels, maybe)
    if last_label == -1:
        first_label = parse_line_answer(first_line, task_labels, maybe)
        if first_label == -1:
            full_label = parse_line_answer(clean_response, task_labels, maybe)
            
            if full_label == -1:
                for line in response_lines:
                    line_label = parse_line_answer(line, task_labels, maybe)
                    if line_label != -1:
                        return line_label 
                return full_label
            else:
                return full_label
        else:
            return first_label
    else:
        return last_label
        
def category_style(category):
    return category.replace(".", "").upper()

def attack_style(attack):
    return ATTACK_NAME[attack]

def parse_code(row, test: CodeSecurityTests):
    extract_result = re.findall(r"```(.*?)```", row.get("response", ""), re.DOTALL)
    if len(extract_result) == 0:
        generated_code = row.get("response", "")
    else:
        generated_code = extract_result[0]

    # From documentation:
    # For autocomplete, we compute "code before" and "code after".
    # "code before" is used as context for the insecure code detector,
    # as an insecure pattern might extend over both the code
    # fed into the LLM and the code returned from the LLM.
    #
    # "code after" is used to compute the BLEU score between the
    # original code and the code returned from the LLM.

    code_before = None
    if test == CodeSecurityTests.AUTOCOMPLETE:
        line_text = row.get("line_text", None)
        origin_code = row.get("origin_code", None)
        if line_text and origin_code:
            code_before = origin_code.split(line_text)[0]
    
    if code_before:
        normalized_code_before = code_before.strip().replace("\r\n", "\n").replace("\r", "\n")       
        normalized_generated_code = generated_code.strip().replace("\r\n", "\n").replace("\r", "\n")       
        if not normalized_generated_code.startswith(normalized_code_before):
            return f"{code_before}\n{generated_code}"
    return generated_code

def clean_response(response: str) -> str:
    cleaned_response = clean_header_footer(response)
    cleaned_response = re.sub(r".*?</think>", "", cleaned_response, flags=re.DOTALL).strip()
    cleaned_response = re.sub(r"<verify>.*?</verify>", "", cleaned_response, flags=re.DOTALL).strip()
    return cleaned_response

def is_no_response(response) -> bool:
    if response is None:
        return True
    normalized = str(response).strip().lower()
    return normalized == "no response" or normalized == "<empty>" or normalized == ""

def mark_no_response_as_na(df, response_col="response", label_col="label"):
    if response_col in df.columns and label_col in df.columns:
        mask = df[response_col].astype(str).str.strip().str.lower().isin({"no response", "<empty>", ""})
        if is_numeric_dtype(df[label_col]):
            df.loc[mask, label_col] = -1
        else:
            df.loc[mask, label_col] = "N/A"
    return df

def compute_no_response_stats(responses: Sequence[Any]) -> Tuple[int, int, float, List[int]]:
    total = len(responses)
    empty_indices: List[int] = [idx for idx, resp in enumerate(responses) if is_no_response(resp)]
    empty_count = len(empty_indices)
    ratio = (empty_count / total) if total else 0.0
    return empty_count, total, ratio, empty_indices

def clean_header_footer(response: str) -> str:
    header_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>"
    footer_pattern = r"<\|endoftext\|>"
    turn_end_pattern = r"<end_of_turn>"

    text = html.unescape(response)
    text = re.sub(header_pattern, "", text)
    text = re.sub(footer_pattern, "", text)
    text = re.sub(turn_end_pattern, "", text)
    return text.strip()

def load_model(hf_model_path, hf_token, models_dir="/opt/airflow/models"):
    """
    Checks if the model exists locally in the specified directory.
    If not, downloads the model using a subprocess.

    Args:
        hf_model_path (str): Hugging Face model path.
        hf_token (str): Hugging Face token for downloading the model.
        models_dir (str): Directory where models should be cached.

    Returns:
        str: Local path to the model.
    """
    cached_model_name = hf_model_path.replace("/", "_")
    cached_model_path = os.path.join(models_dir, cached_model_name)

    if not os.path.exists(cached_model_path):
        print(f"Model {hf_model_path} not found locally. Downloading...")
        try:
            result = subprocess.run([
                "python",
                "/opt/airflow/scripts/download-model.py",
                "--model", hf_model_path,
                "--hf-token", hf_token,
                "--output", cached_model_path
            ])
        
            if result.returncode != 0:
                raise RuntimeError(f"Failed to download the model {hf_model_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(e.stderr)    

    try:
        print(f"Model {cached_model_path} being verified...")

        subprocess.run([
            "python",
            "/opt/airflow/scripts/verify_download.py",
            cached_model_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Model verification failed for {hf_model_path}: {e.stderr}")

    return cached_model_path

def validate_openai_api(base_url, api_key, model_name):
    if "azure" in base_url:
        return
    
    client = OpenAI(base_url=base_url, api_key=api_key)

    try:
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
            temperature=0.0,
        )
    except Exception as e:
        print("Failed to validate API credentials.")

def summarize_model(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    summary = {}
    field_synonyms = {
        "architectures": [],
        "hidden_size": ["n_embd"],
        "intermediate_size": ["n_inner"],
        "max_position_embeddings": ["n_positions"],
        "model_type": [],
        "num_hidden_layers": ["n_layer"],
        "transformers_version": [],
        "vocab_size": []
    }

    for field, synonyms in field_synonyms.items():
        synonyms.append(field)
        for field_synonym in synonyms:
            if field_synonym in config:
                summary[camel_case(field)] = config[field_synonym]

    return summary

def safe_shutdown_vllm(llm):
    if llm is None:
        print("No local vLLM model found for cleanup")
        return
    
    engine = None
    if hasattr(llm, "llm_engine"):
        engine = llm
    elif hasattr(llm, "llm") and hasattr(llm.llm, "llm_engine"):
        engine = llm.llm

    if engine is None:
        print("No local vLLM engine found for cleanup")
        return

    with contextlib.suppress(Exception):
        destroy_model_parallel()
        destroy_distributed_environment()

    with contextlib.suppress(AttributeError):
        del engine.llm_engine.model_executor
    with contextlib.suppress(AttributeError):
        del engine

    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()

    gc.collect()
    torch.cuda.empty_cache()

    with contextlib.suppress(Exception):
        ray.shutdown()

    print("Successfully shut down vLLM")
