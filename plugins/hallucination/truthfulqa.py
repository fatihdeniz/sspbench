import os
import re
import json
import pandas as pd
from hallucination.base_test import BaseTest
from query import generate_query
import utils

BENCHMARKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../services/hallucination"))

# TASK_MESSAGE for TruthfulQA
TASK_MESSAGE = {
    "truthfulqa": {
        "query": {
            "system": "{instruction}",
            "user": "{prompt}"
        }
    }
}

def parse_truthfulqa_output(response):
    try:
        if isinstance(response, str):
            response = response.replace("\\", "")
            match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response, re.DOTALL)
            if match:
                response = match.group(1)
        if isinstance(response, str):
            parsed_output = json.loads(response)
            answer = parsed_output.get("answer", "").strip()[0]
            reasoning = parsed_output.get("reasoning", "")
            response = f"{reasoning} Answer: [{answer}]"
        else:
            answer = response.get("answer", "").strip()[0] if response.get("answer") else None
            reasoning = response.get("reasoning", "")
            response = f"{reasoning} Answer: [{answer}]"
        if answer:
            return {"answer": answer, "response": response}
    except Exception as e:
        pass
    reasoning_match = re.search(r'"?reasoning"?\**?:\s*"?(.+?)"?\s*(?:\s+"?answer"?\s*|$)', response, re.IGNORECASE | re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else response
    patterns = [
        r'(?i)\b(?:the\s+correct\s+)?answer\s+is\s+["\']?\**([A-Z])\**["\']?\b',
        r'(?i)\b(?:the\s+correct\s+)?answer\s*:\s*["\']?\**([A-Z])\**["\']?\b',
        r'(?i)^\s*["\']?\**([A-Z])\**["\']?\.\s',
        r'(?i)\banswer["\']?\s*:\s*["\']?\**([A-Z0-9])\**["\']?\b',
        r'"answer"\s*:\s*["\']?.*?\(\s*(?:[^A-Z]*?)?([A-Z])\s*\)["\']'
    ]
    answer = None
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            answer = match.group(1).strip()
            break
    if not answer:
        last_resort_match = re.search(r'\b([A-F])\b', response)
        if last_resort_match:
            answer = last_resort_match.group(1)
                
    if answer:
        response = f"{reasoning} Answer: [{answer}]"
        return {"answer": answer, "response": response}
    return {"answer": "Answer not found.", "response": response}

class TruthfulQATest(BaseTest):
    def load_data(self):
        if len(self.indices_to_process) > 0:
            truthfulqa_instruction_path = os.path.join(BENCHMARKS_DIR, "instructions", "truthfulqa_mc1.txt")
            with open(truthfulqa_instruction_path, "r") as f:
                truthfulqa_instruction = f.read()
            
            self.queries = self.df.loc[self.indices_to_process].apply(
                lambda x: generate_query(
                    x, TASK_MESSAGE["truthfulqa"], self.args, self.llm,
                    instruction=truthfulqa_instruction, prompt_field="prompt"
                ), axis=1
            ).tolist()
    
    def post_process(self):
        self.df.loc[self.indices_to_process, "raw_response"] = self.responses
        self.df.loc[self.indices_to_process, "clean_response"] = (
            self.df.loc[self.indices_to_process, "raw_response"]
            .apply(utils.clean_response)
        )
        
        parsed_results = self.df.loc[self.indices_to_process, "clean_response"].apply(parse_truthfulqa_output)
        self.df.loc[self.indices_to_process, "parsed_" + self.args.output_field] = parsed_results.apply(lambda x: x["answer"])
        self.df.loc[self.indices_to_process, self.args.output_field] = parsed_results.apply(lambda x: x["response"])
        
    def get_output_columns(self):
        return [self.args.output_field, "raw_response", "conversation", "clean_response", 
                "parsed_" + self.args.output_field]