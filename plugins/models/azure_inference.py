from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_result

from llm_config import LLMConfig
from models.llm_base import LLMBase

class AzureInferenceLLM(LLMBase):
    def __init__(self, model: LLMConfig, generation_config=None):
        if model.type != "azure":
            raise ValueError("Azure Inference requires an Azure-based LLMConfig.")
        if not model.api_url:
            raise ValueError("API URL is missing in the LLMConfig.")
        
        client_params = {
            "endpoint": model.api_url,
            "credential": AzureKeyCredential(model.api_token),
        }
        if model.api_version:
            client_params["api_version"] = model.api_version
        
        self.client = ChatCompletionsClient(**client_params)
        
        self.model_name = model.model
        self.batch = False
        self.working_params = None
        self._role_support_cached = None

        super().__init__(generation_config=generation_config)

    def load_model(self):
        pass

    def generate(self, batch, sampling_params=None):
        batch = self.normalize_batch(batch)
        
        responses = []
        for messages in batch:
            try:
                responses.append(self._generate_single(messages, sampling_params))
            except Exception as e:
                print(f"Azure inference API call failed: {repr(e)}")
                responses.append("<EMPTY>")
                # raise Exception(f"API call failed: {e}")
        
        return responses
    
    
    @retry(
        retry=retry_if_exception_type((Exception)) | retry_if_result(lambda x: x == ""),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=3, min=3, max=30),
    )
    def _generate_single(self, messages, sampling_params=None):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        request_payload = {"messages": messages}
        if self.working_params is not None:
            request_payload.update(self.working_params)
        elif sampling_params:
            request_payload["temperature"] = sampling_params.temperature
            request_payload["max_tokens"] = sampling_params.max_tokens
            request_payload["top_p"] = sampling_params.top_p
            request_payload["top_k"] = sampling_params.top_k
            request_payload["presence_penalty"] = sampling_params.presence_penalty

        try:
            response = self.client.complete(model=self.model_name, **request_payload)
            if self.working_params is None:
                self.working_params = {k: v for k, v in request_payload.items() if k != "messages"}
            result = response.choices[0].message.content.strip()

            if result == "":
                raise ValueError("Received empty response from API. Retrying...")

            return result
        except HttpResponseError as e:            
            error_message = str(e)
            print(f"API Error: {error_message}")
            if "Unsupported parameter: 'max_tokens'" in error_message:
                request_payload.pop("max_tokens", None)
                request_payload["max_output_tokens"] = sampling_params.max_tokens

            try:
                response = self.client.complete(model=self.model_name, **request_payload)
                if self.working_params is None:
                    self.working_params = {k: v for k, v in request_payload.items() if k != "messages"}
                result = response.choices[0].message.content.strip()
                
                if result == "":
                    raise ValueError("Received empty response from API after retry. Retrying...")

                return result
            except HttpResponseError as final_error:
                print(f"API Error: {str(final_error)}")
                return "<EMPTY>"
            
    def check_role_support(self, role):
        if self._role_support_cached is not None:
            return self._role_support_cached

        conversation = [{"role": role, "content": "Test message"}]
        
        try:
            self.client.complete(model=self.model_name, messages=conversation)
            self._role_support_cached = True
        except HttpResponseError as e:
            print(f"Role '{role}' not supported. API Error: {e}")
            self._role_support_cached = False
        except Exception as e:
            print(f"Unexpected error while checking role support: {e}")
            self._role_support_cached = False

        return self._role_support_cached
    
    def _apply_generation_config(self) -> None:
        if not self._generation_config:
            return

        self.working_params = None
        test_request = {
            "messages": [{"role": "user", "content": "Test message"}]
        }
        test_request.update(self._generation_config)

        try:
            self.client.complete(model=self.model_name, **test_request)
            self.working_params = {k: v for k, v in test_request.items() if k != "messages"}
        except HttpResponseError as e:
            error_message = str(e)
            if "Unsupported parameter: 'max_tokens'" in error_message:
                test_request.pop("max_tokens", None)
                test_request["max_output_tokens"] = self._generation_config.get("max_tokens", 1024)

            try:
                self.client.complete(model=self.model_name, **test_request)
                self.working_params = {k: v for k, v in test_request.items() if k != "messages"}
            except HttpResponseError as final_error:
                print(f"[Error] Failed to apply generation config even after fallback: {final_error}")