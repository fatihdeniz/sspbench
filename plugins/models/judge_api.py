import requests

from llm_config import LLMConfig
from models.llm_base import LLMBase

class JudgeApiLLM(LLMBase):
    def __init__(self, model: LLMConfig, generation_config=None, **kwargs):
        if model.type != "api":
            raise ValueError("JudgeLLM requires an API-based LLMConfig.")
        if not model.api_url:
            raise ValueError("API URL is missing in the LLMConfig.")
        
        self.api_url = model.api_url
        self.api_token = model.api_token
        super().__init__(generation_config=generation_config, **kwargs)

    def load_model(self):
        pass

    def generate(self, batch, sampling_params=None):
        batch = self.normalize_batch(batch)

        sampling_params = sampling_params or self._default_sampling_params
        
        if not isinstance(batch, list) or not batch:
            raise ValueError("Batch should be a non-empty list of prompts.")

        headers = {'Content-Type': 'application/json'}
        if self.api_token:
            headers['Authorization'] = f"Bearer {self.api_token}"
        
        payload = {"prompts": batch,
            "max_tokens":sampling_params.max_tokens,
            "temperature":sampling_params.temperature,
            "top_p":sampling_params.top_p,
            "presence_penalty":sampling_params.presence_penalty,
            "repetition_penalty":sampling_params.repetition_penalty
            }
        payload = {k: v for k, v in payload.items() if v is not None}
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Judge API call failed with status code {response.status_code}, Response: {response.text}")
