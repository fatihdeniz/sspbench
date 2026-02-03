import requests

from models.llm_base import LLMBase

class ApiLLM(LLMBase):
    def __init__(self, api_url, api_key):
        self.api_key = api_key
        self.api_url = api_url
        super().__init__()

    def load_model(self):
        pass

    def generate(self, batch, sampling_params):
        if isinstance(batch, str):
            batch = [batch]
            
        headers = {'Content-Type': 'application/json'}
        if self.api_token:
            headers['Authorization'] = f"Bearer {self.api_token}"
            
        responses = []
        for prompt in batch:
            data = {
                'prompt': prompt,
                'max_tokens': sampling_params.max_tokens,
                'temperature': sampling_params.temperature,
                'repetition_penalty': sampling_params.repetition_penalty
            }
            response = requests.post(self.api_url, headers=headers, json=data)
            if response.status_code == 200:
                responses.append(response.json()['choices'][0]['text'].strip())
            else:
                raise Exception(f"API call failed with status code {response.status_code}")
        return responses
