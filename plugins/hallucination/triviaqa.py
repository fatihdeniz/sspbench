import pandas as pd
from hallucination.base_test import BaseTest
from hallucination.simpleqa import qa_judge_mapping

from query import generate_query
import utils

# TASK_MESSAGE for TriviaQA
TASK_MESSAGE = {
    "triviaqa": {
        "query": {
            "system": "You are a helpful assistant.",
            "user": "{prompt}"
        },
        "judge": {
            "system": (
                "You are an expert QA judge. Your job is to evaluate a predicted answer "
                "against a gold (correct) answer and its possible aliases.\n"
                "Assign exactly one of the following grades: [\"CORRECT\", \"INCORRECT\", \"NOT_ATTEMPTED\"].\n\n"
                "- CORRECT: if the predicted answer contains a semantically equivalent answer to the gold answer "
                "or any of the provided aliases, even if the wording is different or additional information is included. "
                "Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.\n"
                "- INCORRECT: if it does not semantically match the gold answer or any of its aliases alias.\n"
                "- NOT_ATTEMPTED: if the model refuses to answer the question for any reason, gives no response, "
                "or provides information that is entirely irrelevant or non-factual."
            ),
            "user": (
                "Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT_ATTEMPTED. "
                "Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.\n"
                "Question: {prompt}\n"
                "Gold (Correct) Answer: {target}\n"
                "Gold Answer Aliases: {target_type}\n\n"
                "```\n"
                "Predicted Answer: {response}\n"
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

class TriviaQATest(BaseTest):
    def load_data(self):
        if len(self.indices_to_process) > 0:
            self.queries = self.df.loc[self.indices_to_process].apply(
                lambda x: generate_query(
                    x, TASK_MESSAGE["triviaqa"], self.args, self.llm,
                    prompt_field="question", target_type_field="target_type",
                    response_field=self.args.response_field
                ), axis=1
            ).tolist()

        if "pre_response" in self.df.columns:
            for i, conv in enumerate(self.queries):
                conv.add_message("assistant", self.df.iloc[i]["pre_response"])
            
        print(f"Loaded {len(self.queries)} queries for TriviaQA test. Sample query: {self.queries[0].to_list()}")

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
            return [self.args.output_field, 'judge_raw', 'judge_conversation']
        else:
            return [self.args.output_field, 'response_raw', 'conversation']