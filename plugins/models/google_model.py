from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from llm_config import LLMConfig
from models.llm_base import LLMBase

class GoogleLLM(LLMBase):
    def __init__(self, model: LLMConfig, generation_config=None):
        if model.type != "google":
            raise ValueError("Google Gemini API requires a Google-based LLMConfig.")
        if not model.api_token:
            raise ValueError("API Token is missing in the LLMConfig.")

        self.client = genai.Client(api_key=model.api_token)
        self.model_name = model.model
        self.batch = False
        super().__init__(generation_config=generation_config)

    def load_model(self):
        pass

    def generate(self, batch, sampling_params=None):
        batch = self.normalize_batch(batch)

        sampling_params = sampling_params or self._default_sampling_params
        
        responses = []
        
        config_params = {
            "max_output_tokens":sampling_params.max_tokens,
            "temperature":sampling_params.temperature,
            "top_p":sampling_params.top_p,
            "presence_penalty":sampling_params.presence_penalty,
            "repetition_penalty":sampling_params.repetition_penalty
        }
        config_params = {k: v for k, v in config_params.items() if v is not None}

        for messages in batch:
            try:
                responses.append(self._generate_single(messages, config_params))
            except Exception as e:
                responses.append("<EMPTY>")
                print(f"Google Gemini API call failed: {repr(e)}")

        return responses

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _generate_single(self, messages, config_params):
        contents = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                contents.append(msg["content"])
        
        if system_instruction:
            config_params["system_instruction"] = system_instruction
        
        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(**config_params),
            contents=contents
        )
        return response.text.strip() if response.text else "No response"

