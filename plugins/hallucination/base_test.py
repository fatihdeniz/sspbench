import os
import pandas as pd
from abc import ABC, abstractmethod

from models.llm_base import EmptyModelResponseError
from utils import compute_no_response_stats
from cache_utils import load_cache


class BaseTest(ABC):
    def __init__(self, args, llm, sampling_params):
        self.args = args
        self.llm = llm
        self.sampling_params = sampling_params
        self.queries = []
        self.responses = []
        self.df = None  
        self.indices_to_process = []

    @abstractmethod
    def load_data(self):
        """Load input data (or prompts) and build queries."""
        pass
    
    @abstractmethod
    def get_output_columns(self):
        pass

    def run_inference(self, queries, sampling_params):
        """Generate responses in batches using the LLM."""
        responses, conversations = [], []
        for i in range(0, len(queries), self.args.batch_size):
            batch = queries[i:i + self.args.batch_size]
            batch_conversation, batch_responses = self.llm.generate_conversation(batch, sampling_params)
            responses.extend(batch_responses)
            conversations.extend(batch_conversation)
        return conversations, responses

    @abstractmethod
    def post_process(self):
        """Process the responses (e.g. cleaning, judging, parsing) and build the final DataFrame."""
        pass

    def save_results(self):
        self.df.to_json(self.args.output, orient="records", lines=True)
        
    def load_cache_if_exists(self):
        if os.path.exists(self.args.input) and os.path.exists(self.args.output):
            self.df, self.indices_to_process = load_cache(
                self.args.input,
                self.args.output,
                output_cols=self.get_output_columns()
            )
            return True
        return False

    def run(self):
        cache_loaded = self.load_cache_if_exists()
        
        if not cache_loaded and os.path.exists(self.args.input):
            self.df = pd.read_json(self.args.input, lines=True)
            self.indices_to_process = self.df.index.tolist()
        
        if self.df is not None and len(self.indices_to_process) == 0:
            print(f"All responses already processed. Results available at {self.args.output}")
            return
        
        print(f"Processing {len(self.indices_to_process)} rows")
        
        self.load_data()
        conversations, self.responses = self.run_inference(self.queries, self.sampling_params)
        
        if conversations and len(conversations) == len(self.df):
            conversation_col = "judge_conversation" if self.args.judge else "conversation"
            self.df.loc[self.indices_to_process, conversation_col] = pd.Series(
                conversations, index=self.indices_to_process
            )
        elif self.args.judge:
            n_rows = len(self.indices_to_process)
            n_samples = len(conversations) // n_rows if n_rows > 0 else 0

            for i in range(n_samples):
                start = i * n_rows
                end = (i + 1) * n_rows
                self.df.loc[self.indices_to_process, f"judge_conversation_{i+1}"] = pd.Series(
                    conversations[start:end], index=self.indices_to_process
                )

        self.post_process()
        self.save_results()
        
        empty_count, total_count, empty_ratio, _ = compute_no_response_stats(self.df[self.get_output_columns()[0]])

        if total_count and empty_ratio > 0.01:
            raise EmptyModelResponseError(
                f"Model returned {empty_count} empty responses out of {total_count} prompts "
                f"({empty_ratio:.2%})."
            )
        
