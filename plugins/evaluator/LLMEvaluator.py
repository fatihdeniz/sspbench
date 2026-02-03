from abc import abstractmethod
from typing import List
from conversation import Conversation
from models.llm_base import LLMBase
from evaluator import Evaluator

class LLMEvaluator(Evaluator):
    
    def __init__(self, eval_model: LLMBase):
        super().__init__(eval_model)
        if eval_model is None:
            raise ValueError("LLMEvaluator requires an eval_model")
    
    @abstractmethod
    def _evaluate(self, instance, **kwargs):
        return NotImplementedError()
    
    def make_llm_request(self, queries: List[Conversation], sampling_params=None):
        responses, conversations = [], []
        for i in range(0, len(queries), self.args.batch_size):
            batch = queries[i:i + self.args.batch_size]
            batch_conversation, batch_responses = self.eval_model.generate_conversation(batch, sampling_params)
            responses.extend(batch_responses)
            conversations.extend(batch_conversation)
        return conversations, responses