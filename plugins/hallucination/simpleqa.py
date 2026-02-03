import os
import re
import pandas as pd
from hallucination.base_test import BaseTest
from query import generate_query
import utils

BENCHMARKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../services/hallucination"))

# TASK_MESSAGE for SimpleQA
TASK_MESSAGE = {
    "simpleqa": {
        "query": {
            "system": "You are a helpful assistant.",
            "user": "{prompt}"
        },
        "judge": {
            "system": (
                "{instruction}\n"
                "Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either [\"CORRECT\", \"INCORRECT\", \"NOT_ATTEMPTED\"]."
                "The predicted answer should be considered CORRECT if it contains an answer that is semantically equivalent to the gold target even if the wording is different."
            ),
            "user": (
                "Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT_ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.\n"
                "```\n"
                "Question: {prompt}\n"
                "Gold target: {target}\n"
                "Predicted answer: {response}\n"
                "```\n\n"
                "Grade the predicted answer of this new question as one of:\n"
                "A: CORRECT\n"
                "B: INCORRECT\n"
                "C: NOT_ATTEMPTED\n\n"
                "Just return the letters \"A\", \"B\", or \"C\", with no text around it."
            )
        }
    }
}

def qa_judge_mapping(judge_response):
    judge_response_upper = judge_response.upper()
    if "NOT_ATTEMPTED" in judge_response_upper:
        return "Not Attempted"
    elif "INCORRECT" in judge_response_upper:
        return "Incorrect"
    elif "CORRECT" in judge_response_upper:
        return "Correct"
    match = re.search(r"(A|B|C)", judge_response)
    if match:
        if match.group(0) == "A":
            return "Correct"
        elif match.group(0) == "B":
            return "Incorrect"
        elif match.group(0) == "C":
            return "Not Attempted"
    return "Not Attempted"

class SimpleQATest(BaseTest):
    def load_data(self):
        if len(self.indices_to_process) > 0:
            self.queries = self.df.loc[self.indices_to_process].apply(
                lambda x: generate_query(
                    x, TASK_MESSAGE["simpleqa"], self.args, self.llm,
                    response_field=self.args.response_field
                ), axis=1
            ).tolist()
            
    def post_process(self):
        if self.args.judge:
            self.df.loc[self.indices_to_process, 'judge_raw'] = self.responses
            self.df.loc[self.indices_to_process, self.args.output_field] = (
                self.df.loc[self.indices_to_process, 'judge_raw']
                .apply(utils.clean_response)
                .apply(qa_judge_mapping)
            )
        else:
            self.df.loc[self.indices_to_process, 'response_raw'] = self.responses
            self.df.loc[self.indices_to_process, self.args.output_field] = (
                self.df.loc[self.indices_to_process, 'response_raw']
                .apply(utils.clean_response)
            )

    def get_output_columns(self):
        if self.args.judge:
            return [self.args.output_field, "judge_raw", "judge_conversation"] 
        return [self.args.output_field, "response_raw", "conversation"]
