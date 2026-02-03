import os
import re
import json
import copy
import pandas as pd
import numpy as np
from vllm import SamplingParams
from hallucination.base_test import BaseTest
from query import generate_query
import utils

BENCHMARKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../services/hallucination"))

# TASK_MESSAGE for SelfCheckGPT
TASK_MESSAGE = {
    "selfcheckgpt": {
        "query": {
            "system": "You are a helpful assistant.",
            "user": "{prompt}"
        },
        "judge": {
            "system": "{instruction}",
            "user": "\n\n#Question#: {prompt}\n\n#Answer#: {response}\n\n#Your Judgement#: "
        },
        "judge_2": {
            "system": "{instruction}",
            "user": "\n\n#Factual Sentences#: {target}\n#Context#: {response}\n\n#Your Judgement#: "
        },
        "judge_fact": {
            "system": "{instruction}",
            "user": "\n\n#Question#: {prompt} \n\n#Answer#: {response} \n\n#Factual Statements#:"
        }
    }
}

def selfcheckgpt_rta_judge(judge_response):
    try:
        if isinstance(judge_response, str):
            judge_response = judge_response.replace("\\", "")
            match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", judge_response, re.DOTALL)
            if match:
                judge_response = match.group(1)
        if isinstance(judge_response, str):
            parsed_response = json.loads(judge_response)
        else:
            parsed_response = judge_response
        category = parsed_response.get("category", "").strip().lower()
        if "factual" in category:
            return "Certain"
        else:
            return "Uncertain"
    except (json.JSONDecodeError, AttributeError):
        pass
    judge_response_lower = judge_response.lower() if isinstance(judge_response, str) else str(judge_response).lower()
    if "uncertain" in judge_response_lower:
        return "Uncertain"
    else:
        return "Certain"

def selfcheckgpt_aggreement_judge(output: str) -> int:
    try:
        if isinstance(output, str):
            processed_str = output.replace("\\", "")
            match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", processed_str, re.DOTALL)
            if match:
                processed_str = match.group(1)
            data = json.loads(processed_str)
        else:
            data = output

        score_str = data.get("agreement_score")
        
        if isinstance(score_str, str) and "/" in score_str:
            score_value = float(score_str.split('/')[0].strip())
            return int(score_value * 10)
        
        return 50
        
    except (json.JSONDecodeError, AttributeError, ValueError, IndexError, TypeError):
        return 50

class SelfCheckGPTTest(BaseTest):
    SAMPLE_COUNT = 10
    def load_data(self):
        self.df = pd.read_json(self.args.input, lines=True)
        if self.args.judge:
            # Generate RTA queries for refusal to answer check
            rta_instructions_path = os.path.join(BENCHMARKS_DIR, "instructions", "selfcheckgpt-rta.txt")
            with open(rta_instructions_path, "r") as f:
                rta_instruction = f.read()
            rta_queries = self.df.apply(lambda x: generate_query(
                x,
                TASK_MESSAGE["selfcheckgpt"],
                self.args,
                self.llm,
                instruction=rta_instruction,
                judge_field="judge",
            ), axis=1).tolist()
            _, rta_responses = self.run_inference(rta_queries, self.sampling_params)
            self.df["rta_judge_raw"] = rta_responses
            self.df["rta_judge"] = self.df["rta_judge_raw"].apply(selfcheckgpt_rta_judge)
            
            # Extract facts
            fact_instructions_path = os.path.join(BENCHMARKS_DIR, "instructions", "selfcheckgpt-fact.txt")
            with open(fact_instructions_path, "r") as f:
                fact_instruction = f.read()
            
            fact_extraction_queries = self.df.apply(lambda x: generate_query(
                x,
                TASK_MESSAGE["selfcheckgpt"],
                self.args,
                self.llm,
                instruction=fact_instruction,
                judge_field="judge_fact",
            ), axis=1).tolist()
            fact_conversations, fact_extraction_responses = self.run_inference(fact_extraction_queries, self.sampling_params)
            self.df["extract_fact_conversation"] = fact_conversations
            self.df["extracted_facts"] = [utils.clean_response(resp) for resp in fact_extraction_responses]
            
            # Generate uncertainty queries for multiple samples
            uncertainty_instructions_path = os.path.join(BENCHMARKS_DIR, "instructions", "selfcheckgpt.txt")
            with open(uncertainty_instructions_path, "r") as f:
                uncertainty_instruction = f.read()
            self.queries = []
            for _ in range(SelfCheckGPTTest.SAMPLE_COUNT):
                queries_i = self.df.apply(lambda x: generate_query(
                    x,
                    TASK_MESSAGE["selfcheckgpt"],
                    self.args,
                    self.llm,
                    instruction=uncertainty_instruction,
                    judge_field="judge_2",
                    response_field=f"sample_{_ + 1}",
                    target_field="extracted_facts"
                ), axis=1).tolist()
                self.queries.extend(queries_i)
        else:
            self.queries = self.df.apply(lambda x: generate_query(x, TASK_MESSAGE["selfcheckgpt"], self.args, self.llm), axis=1).tolist()
            params_hightemp = SamplingParams(temperature=1, repetition_penalty=1.1, max_tokens=1024)
            repeated_queries = [copy.deepcopy(c) for c in self.queries for _ in range(SelfCheckGPTTest.SAMPLE_COUNT)]

            _, sample_responses = self.run_inference(repeated_queries, params_hightemp)
            
            grouped_responses = [[] for _ in self.queries]
            for i, response in enumerate(sample_responses):
                original_index = i // SelfCheckGPTTest.SAMPLE_COUNT
                grouped_responses[original_index].append(utils.clean_response(response))

            for j in range(SelfCheckGPTTest.SAMPLE_COUNT):
                self.df[f"sample_{j + 1}"] = [group[j] if j < len(group) else None for group in grouped_responses]
            
    
    def post_process(self):
        if self.args.judge:
            cleaned_responses = [utils.clean_response(r) for r in self.responses]
            judge_scores = [selfcheckgpt_aggreement_judge(r) for r in cleaned_responses]
            judge_response_df = pd.DataFrame(np.array(cleaned_responses).reshape(-1, SelfCheckGPTTest.SAMPLE_COUNT),
                                             columns=[f"judge_response_{j + 1}" for j in range(SelfCheckGPTTest.SAMPLE_COUNT)])
            judge_scores_df = pd.DataFrame(np.array(judge_scores).reshape(-1, SelfCheckGPTTest.SAMPLE_COUNT),
                                           columns=[f"score_{j + 1}" for j in range(SelfCheckGPTTest.SAMPLE_COUNT)])
            self.df = pd.concat([self.df, judge_response_df, judge_scores_df], axis=1)
            self.df[self.args.output_field] = judge_scores_df.mean(axis=1).astype(int)
        else:
            self.df[self.args.output_field] = self.responses
            self.df[self.args.output_field] = self.df[self.args.output_field].apply(utils.clean_response)

    def get_output_columns(self):
        return [self.args.output_field]