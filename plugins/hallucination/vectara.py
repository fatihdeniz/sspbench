import os
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from hallucination.base_test import BaseTest
from query import generate_query
import utils
from config import Config


BENCHMARKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../services/hallucination"))

# TASK_MESSAGE for Vectara
TASK_MESSAGE = {
    "vectara": {
        "query": {
            "system": ("You are a chat bot answering questions using data."
                    "You must stick to the answers provided solely by the text in the passage provided."
                    "You are asked the question 'Provide a concise summary of the given passage, covering the core pieces of information described.'"
                    ),
            "user": ("Provide a concise summary of the given passage, covering the core pieces of information described.\n"
                "PASSAGE: {prompt}\n\n"
                "SUMMARY: " 
                )
        }
    }
}

class VectaraJudge:
    def __init__(self, model_name='vectara/hallucination_evaluation_model', tokenizer_name='google/flan-t5-base'):
        
        model_path = utils.load_model(model_name, Config.HF_TOKEN)
        tokenizer_path = utils.load_model(tokenizer_name, Config.HF_TOKEN)
        
        self.prompt_template = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
        self.classifier = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True),
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_path),
            trust_remote_code=True
        )

    def predict(self, pairs):
        try:
            input_pairs = [self.prompt_template.format(text1=pair[0], text2=pair[1]) for pair in pairs]
            full_scores = self.classifier(input_pairs, top_k=None)
            
            # Extract the scores for the 'consistent' label, other label is 'hallucinated'
            # for details refer to https://huggingface.co/vectara/hallucination_evaluation_model
            consistent_scores = [
                int(100 * score_dict["score"])
                for score_for_both_labels in full_scores
                for score_dict in score_for_both_labels
                if score_dict["label"].lower() == "consistent"
            ]
            
            labels = [ max(score_for_both_labels, key=lambda x: x["score"])["label"] for score_for_both_labels in full_scores]
        
            return consistent_scores, labels
        except Exception as e:
            print(f"Vectara judge failed: {e}")
            return [0.0] * len(pairs), ["Unknown"] * len(pairs)

class VectaraTest(BaseTest):
    def load_data(self):
        if self.args.judge:
            judge = VectaraJudge()
            self.scores, self.labels = judge.predict(zip(self.df.loc[self.indices_to_process,self.args.input_field].tolist(), 
                                                        self.df.loc[self.indices_to_process,self.args.output_field].tolist()))
        else:
            self.queries = self.df.loc[self.indices_to_process].apply(lambda x: generate_query(
                x, TASK_MESSAGE["vectara"], self.args, self.llm), axis=1).tolist()
    
    def post_process(self):
        if self.args.judge:
            self.df.loc[self.indices_to_process, "judge_score"] = self.scores
            self.df.loc[self.indices_to_process, "raw_label"] = self.labels
            self.df.loc[self.indices_to_process, "label"] = self.labels
        else:
            self.df.loc[self.indices_to_process, "raw_response"] = self.responses
            self.df.loc[self.indices_to_process, self.args.output_field] = self.df["raw_response"].apply(utils.clean_response)

    def get_output_columns(self):
        if self.args.judge:
            return ["judge_score", 'raw_label', 'label', "judge_conversation"]
        else:
            return [self.args.output_field, 'raw_response', 'conversation']