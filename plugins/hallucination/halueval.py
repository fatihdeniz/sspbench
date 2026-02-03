import os
import random
import pandas as pd
from vllm import SamplingParams
from hallucination.base_test import BaseTest
from query import generate_query
import utils

random.seed(42)

# TASK_MESSAGE for HaluEval tests (QA, Dialog, Summary)
TASK_MESSAGE = {
    "halueval-qa": {
        "query": {
            "system": "{instruction}",
            "user": "\n\n#Question#: {prompt}\n#Answer#: {target}\n#Your Judgement#: "
        }
    },
    "halueval-dialog": {
        "query": {
            "system": "{instruction}",
            "user": "\n\n#Dialogue History#: {prompt}\n#Response#: {target}\n#Your Judgement#: "
        }
    },
    "halueval-summary": {
        "query": {
            "system": "{instruction}",
            "user": "#Document#: {prompt}\n#Summary#: {target}\n#Your Judgement#: "
        }
    }
}

# HALUEVAL_TASKS with paths, instructions, and field mappings.
BENCHMARKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../services/hallucination"))

HALUEVAL_TASKS = {
    "halueval-qa": {
        "name": "QA",
        "description": "Question Answering",
        "prompts": f"{BENCHMARKS_DIR}/prompts/halueval-qa.jsonl",
        "instruction": f"{BENCHMARKS_DIR}/instructions/halueval-qa.txt",
        "fields": {
            "prompt": "question", 
            "hallucinated_prompt": "hallucinated_answer", 
            "right_prompt": "right_answer",
            "prompt_template": (
                "You are a hallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge. \n\n" 
                "Question: {prompt}\n\n"
                "Answer: {target}")
        },
    }, 
    "halueval-dialog": {
        "name": "Dialog",
        "description": "Dialog",
        "prompts": f"{BENCHMARKS_DIR}/prompts/halueval-dialog.jsonl",
        "instruction": f"{BENCHMARKS_DIR}/instructions/halueval-dialog.txt",
        "fields": {
            "prompt": "dialogue_history", 
            "hallucinated_prompt": "hallucinated_response", 
            "right_prompt": "right_response",
            "prompt_template": (
                "You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information.\n\n"
                "Dialogue History: {prompt}\n\n"
                "Response: {target}")
        },
    }, 
    "halueval-summary": {
        "name": "Summary",
        "description": "Summarization",
        "prompts": f"{BENCHMARKS_DIR}/prompts/halueval-summary-sample.jsonl",
        "instruction": f"{BENCHMARKS_DIR}/instructions/halueval-summary.txt",
        "fields": {
            "prompt": "document", 
            "hallucinated_prompt": "hallucinated_summary", 
            "right_prompt": "right_summary", 
            "prompt_template": (
                "You are a summary judge. You MUST determine if the provided summary contains non-factual or hallucinated information.\n\n"
                "Document: {prompt}\n\n"
                "Summary: {target}")
        },
    },
}

def evaluate_hallucination_answer(response):
    response = response.strip()
    response_lower = response.lower()
    if ("yes" not in response_lower and "no" not in response_lower):
        return response
    
    yes_pos = response_lower.find("yes") if "yes" in response_lower else None
    no_pos  = response_lower.find("no")  if "no" in response_lower else None

    if yes_pos is not None and no_pos is not None:
        if yes_pos < no_pos:
            return "Hallucination"
        else:
            return "Factual"

    if yes_pos is not None:
        return "Hallucination"
    if no_pos is not None:
        return "Factual"

    return response

class HaluEvalTest(BaseTest):
    def __init__(self, args, llm, sampling_params):
        super().__init__(args, llm, SamplingParams(temperature=0, repetition_penalty=1.1, max_tokens=256))
        
    def load_data(self):
        
        if not os.path.exists(self.args.input):
            short_queries = []
            for task_key, task_data in HALUEVAL_TASKS.items():
                df_task = pd.read_json(task_data["prompts"], lines=True)
                with open(task_data["instruction"], "r") as f:
                    instruction = f.read()
                random_choices = random.choices([True, False], k=len(df_task))
                for random_choice, (_, row) in zip(random_choices, df_task.iterrows()):
                    prompt_field = task_data["fields"]["prompt"]
                    hallucinated_field = task_data["fields"]["hallucinated_prompt"]
                    right_field = task_data["fields"]["right_prompt"]
                    if random_choice:
                        selected_field = hallucinated_field
                        ground_truth = "Hallucination"
                    else:
                        selected_field = right_field
                        ground_truth = "Factual"
                    
                    short_query = task_data["fields"]["prompt_template"].format(
                        prompt=row[prompt_field],
                        target=row[selected_field]
                    )
                    short_queries.append({
                        "prompt": short_query,
                        "raw_prompt": row[prompt_field],
                        "selected_answer": row[selected_field],
                        "ground_truth": ground_truth,
                        "target": row[right_field],
                        "category": task_data["name"],
                        "categoryDescription": task_data["description"],
                        "task": task_key
                    })
            df_combined = pd.DataFrame(short_queries)
            df_combined.to_json(self.args.input, orient="records", lines=True)
            
            self.df = df_combined
            self.indices_to_process = df_combined.index.tolist()
        
        instructions = {}
        for task_key, task_data in HALUEVAL_TASKS.items():
            with open(task_data["instruction"], "r") as f:
                instruction = f.read()
            instructions[task_key] = instruction
        
        if len(self.indices_to_process) > 0:
            self.queries = self.df.loc[self.indices_to_process].apply(
                lambda x: generate_query(x, TASK_MESSAGE[x["task"]],
                    self.args, self.llm,
                    instruction=instructions[x["task"]],
                    prompt_field="raw_prompt",
                    target_field="selected_answer",
                ), axis=1).tolist()
                
    def post_process(self):
        self.df.loc[self.indices_to_process, 'raw_response'] = self.responses
        self.df.loc[self.indices_to_process, self.args.output_field] = self.df.loc[self.indices_to_process, 'raw_response'].apply(utils.clean_response)
        self.df.loc[self.indices_to_process, "parsed_" + self.args.output_field] = self.df.loc[self.indices_to_process, self.args.output_field].apply(evaluate_hallucination_answer)

    def get_output_columns(self):
        return [self.args.output_field, "raw_response", "parsed_" + self.args.output_field, "conversation"]