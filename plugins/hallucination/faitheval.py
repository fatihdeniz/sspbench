import os
import json
import random
import re
import string
import pandas as pd
from hallucination.base_test import BaseTest
from query import generate_query
import utils

random.seed(42)

# TASK_MESSAGE for HaluEval tests (QA, Dialog, Summary)
TASK_MESSAGE = {
    "faitheval": {
        "query": {
            "system": "{instruction}",
            "user": "\n\nContext: {target}\nQuestion: {prompt}\nAnswer:"
        }
    }
}

BENCHMARKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../services/hallucination"))

FAITHEVAL_TASKS = {
    "faitheval-unanswerable": {
        "name": "Unanswerable",
        "description": "Unanswerable Questions",
        "prompts": f"{BENCHMARKS_DIR}/prompts/faitheval-unanswerable.jsonl",
        "instruction": f"{BENCHMARKS_DIR}/instructions/faitheval-unanswerable.txt",
        "fields": {
            "prompt": "question",
        }
    },
    "faitheval-inconsistent": {
        "name": "Inconsistent",
        "description": "Inconsistent Information in Context",
        "prompts": f"{BENCHMARKS_DIR}/prompts/faitheval-inconsistent.jsonl",
        "instruction": f"{BENCHMARKS_DIR}/instructions/faitheval-inconsistent.txt",
        "fields": {
            "prompt": "question",
        }
    },
    "faitheval-counterfactual": {
        "name": "Counterfactual",
        "description": "Counterfactual Statements in Context",
        "prompts": f"{BENCHMARKS_DIR}/prompts/faitheval-counterfactual.jsonl",
        "instruction": f"{BENCHMARKS_DIR}/instructions/faitheval-counterfactual.txt",
        "fields": {
            "prompt": "formatted_question",
        }
    }
}

unanswerable_valid_phrases = ['unknown', 'no answer', 'no information', 'not', 'unclear']
inconsistent_valid_phrases = ['conflict', 'multiple answers', 'disagreement', 'inconsistent', 'contradictory', 'contradiction', 'inconsistency', 'two answers', '2 answers', 'conflicting']

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')
    
    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

class FaithEvalTest(BaseTest):
    def load_data(self):
        if not os.path.exists(self.args.input):
            short_queries = []
            for task_key, task_data in FAITHEVAL_TASKS.items():
                df_task = pd.read_json(task_data["prompts"], lines=True)
                
                for _, row in df_task.iterrows():
                    prompt_field = task_data["fields"]["prompt"]
                    short_queries.append({
                        "context": row["context"],
                        "prompt": row[prompt_field],
                        "category": task_data["name"],
                        "categoryDescription": task_data["description"],
                        "answerKey": row.get("answerKey", ""),
                        "task": task_key
                    })
            df_combined = pd.DataFrame(short_queries)
            df_combined.to_json(self.args.input, orient="records", lines=True)
            
            self.df = df_combined
            self.indices_to_process = df_combined.index.tolist()
        
        instructions = {}
        for task_key, task_data in FAITHEVAL_TASKS.items():
            with open(task_data["instruction"], "r") as f:
                instruction = f.read()
            instructions[task_key] = instruction
        
        if len(self.indices_to_process) > 0:
            self.queries = self.df.loc[self.indices_to_process].apply(
                lambda x: generate_query(x, TASK_MESSAGE["faitheval"],
                    self.args, self.llm,
                    instruction=instructions[x["task"]],
                    prompt_field="prompt",
                    target_field="context",
                ), axis=1).tolist()
    
    def parse_json_answer(self, response):
        try:
            if isinstance(response, str):
                response = response.replace("\\", "")
                match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response, re.DOTALL)
                if match:
                    response = match.group(1)
            if isinstance(response, str):
                parsed_output = json.loads(response)
                answer = parsed_output.get("answer", "").strip()[0]
                response = f"Answer: [{answer}]"
            else:
                answer = response.get("answer", "").strip()[0] if response.get("answer") else None
                response = f"Answer: [{answer}]"
            if answer:
                return {"answer": answer, "response": response}
        except Exception as e:
            pass
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
            last_resort_match = re.search(r'\b([A-E])\b', response)
            if last_resort_match:
                answer = last_resort_match.group(1)
            
        if answer:
            response = f"Answer: [{answer}]"
            return {"answer": answer, "response": response}
        return {"answer": "N/A", "response": response}
    
    def evaluate_by_category(self, row):
        row["parsed_response"] = row[self.args.output_field]
        norm_response = normalize_answer(row["parsed_response"])
        if row["category"] == "Unanswerable":
            row["label"] = any([phrase in norm_response for phrase in unanswerable_valid_phrases])
        elif row["category"] == "Inconsistent":
            row["label"] = any([phrase in norm_response for phrase in inconsistent_valid_phrases])
        else:
            parsed_response = self.parse_json_answer(row["parsed_response"])
            row["label"] = parsed_response["answer"].lower() == row["answerKey"].lower()
            row["parsed_response"] = parsed_response["response"]
        return row
    
    def post_process(self):
        self.df.loc[self.indices_to_process, self.args.output_field] = [utils.clean_response(r) for r in self.responses]
        results = self.df.loc[self.indices_to_process].apply(self.evaluate_by_category, axis=1)
        self.df.loc[self.indices_to_process, "parsed_response"] = results["parsed_response"]
        self.df.loc[self.indices_to_process, "label"] = results["label"]

    def get_output_columns(self):
        return [self.args.output_field, "parsed_response", "label", "conversation"]
