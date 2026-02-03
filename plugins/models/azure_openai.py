from openai import AzureOpenAI  

from llm_config import LLMConfig
from models.llm_base import LLMBase

class AzureOpenaiLLM(LLMBase):
    def __init__(self, model: LLMConfig, **kwargs):
        if model.type != "openai":
            raise ValueError("JudgeOpenAiLLM requires an OpenAI-based LLMConfig.")
        if not model.api_url:
            raise ValueError("API URL is missing in the LLMConfig.")
        
        client_params = {
            "azure_endpoint": model.api_url,
            "api_key": model.api_token,
        }
        if model.api_version:
            client_params["api_version"] = model.api_version
        else:
            client_params["api_version"] = "2024-12-01-preview"
            
        self.client = AzureOpenAI(**client_params)
        
        self.model_name = model.model
        self.batch = True
        super().__init__(**kwargs)

    def load_model(self):
        pass

    def generate(self, batch, sampling_params=None):
        batch = self.normalize_batch(batch)
            
        sampling_params = sampling_params or self._default_sampling_params
        
        if self.batch:
            try:
                completion = self.client.completions.create(
                    model=self.model_name,
                    prompt=batch,
                    temperature=sampling_params.temperature,
                    max_tokens=sampling_params.max_tokens,
                    top_p=sampling_params.top_p,
                    presence_penalty=sampling_params.presence_penalty,
                )
                responses = [choice.text.strip() for choice in completion.choices]
                return responses
            except Exception as e:
                self.batch=False
                print(f"Judge batch API call failed: {repr(e)}. Falling back to chat completions...")
                # raise Exception(f"Batch API call failed: {e}")
            
        responses = []
        for messages in batch:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=sampling_params.temperature,
                    max_tokens=sampling_params.max_tokens,
                    top_p=sampling_params.top_p,
                    presence_penalty=sampling_params.presence_penalty,
                )
                responses.append(completion.choices[0].message.content.strip())
            except Exception as e:
                responses.append("<EMPTY>")
                print(f"Judge chat API call failed: {repr(e)}")
                # raise Exception(f"API call failed: {e}")
        
        return responses
    