from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, BadRequestError, NotFoundError, PermissionDeniedError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_not_exception_type
from models.llm_base import LLMBase

CONTENT_POLICY_RESPONSE = "I apologize, but I cannot provide a response to this request due to content policy restrictions."

class OpenaiLLM(LLMBase):
    def __init__(self, base_url, api_key, model_name, generation_config=None, thread_count=8, **kwargs):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.thread_count = thread_count
        self.batch = True
        self.working_params = None
        super().__init__(generation_config=generation_config, **kwargs)

    def load_model(self):
        if not self.working_params:
            self.working_params = self._discover_supported_params(self._default_sampling_params)

    
    def generate(self, batch, sampling_params=None):
        batch = self.normalize_batch(batch)
        
        sampling_params = sampling_params or self._default_sampling_params
        if not self.working_params:
            self.working_params = self._discover_supported_params(sampling_params)
        
        if self.batch:
            # Try sending batches to the API
            try:
                completion = self.client.completions.create(
                    model=self.model_name,
                    prompt=batch,
                    **self.working_params
                )
                responses = [choice.text.strip() for choice in completion.choices]

                return responses
            except NotFoundError:
                print("API does not support batch inference, trying chat inference...")
            except BadRequestError:
                print("Content in batch was blocked by API model, trying chat inference...")
            except PermissionDeniedError:
                print("Permission denied by API provider, trying chat inference...")
            except Exception as e:
                print(f"Batch API call failed ({repr(e)}), trying chat inference...")
        
            self.batch = False
            
        responses = [None] * len(batch)
        def worker(batch_index, messages):
            try:
                return batch_index, self._generate_single(messages)
            except BadRequestError as e:
                if getattr(e, "code", None) == "content_filter":
                    print("Content was blocked by API model: ", messages)
                    return batch_index, CONTENT_POLICY_RESPONSE
                else:
                    raise
            except PermissionDeniedError:
                print("Content was blocked by API provider: ", messages)
                return batch_index, CONTENT_POLICY_RESPONSE
            except Exception as e:
                print(f"API call failed with error: {repr(e)}")
                return batch_index, "<EMPTY>"

        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            futures = [executor.submit(worker, i, messages) for i, messages in enumerate(batch)]
            for future in as_completed(futures):
                i, result = future.result()
                responses[i] = result

        return responses    
    
    @retry(
        retry=retry_if_exception_type(Exception) & retry_if_not_exception_type(PermissionDeniedError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=3, min=3, max=30),
    )
    def _generate_single(self, messages):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.working_params
        )
        # print(f"{self.model_name}:", completion.choices[0].message.content.strip())
        return completion.choices[0].message.content.strip()
    
    
    def _test_param(self, messages, params):
        try:
            self.client.chat.completions.create(model=self.model_name, messages=messages, **params)
            return True
        except Exception:
            return False