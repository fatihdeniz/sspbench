from typing import Optional, Dict, List, Tuple, Any
from vllm import SamplingParams

from utils import clean_header_footer
from conversation import Conversation


class EmptyModelResponseError(RuntimeError):
    """Raised when a model returns an empty or placeholder response."""


class LLMBase:
    EMPTY_RESPONSE_TOLERANCE = 0.01
    
    def __init__(self, model_path=None,
                generation_config: Optional[Dict[str, Any]] = None,
                fail_on_empty_response: bool = True,
                **kwargs):
        self.model_path = model_path
        self.tokenizer = None
        self.conversation_format = "openai"
        self.generation_config = generation_config
        self.system_message = None
        self.fail_on_empty_response = fail_on_empty_response
        
        self._empty_response_count = 0
        self._total_response_count = 0
        
        self.load_model()
    
    def load_model(self):
        raise NotImplementedError("This method must be implemented in subclasses.")

    def generate(self, batch, sampling_params):
        raise NotImplementedError("This method must be implemented in subclasses.")
    
    def check_role_support(self, role):
        """Skip role support for API-based models. Implemented in HuggingFaceLLM."""
        return True
    
    @property
    def generation_config(self) -> Dict:
        return self._generation_config

    @generation_config.setter
    def generation_config(self, config: Dict) -> None:
        self._generation_config = config or {}
        self._apply_generation_config()
        
    def set_system_message(self, message: str) -> None:
        self.system_message = message

    def generate_conversation(self, batch: List[Conversation], sampling_params=None) -> Tuple[List[Dict[str, str]], List[str]]:
        responses = []
        if self.system_message:
            for conv in batch:
                conv.set_system_message(self.system_message)
                
        prompts = [conv.to_prompt(self) for conv in batch]
        chat_inputs = [conv.to_list(include_history=True, llm=self) for conv in batch]

        if hasattr(self, "llm") and hasattr(self.llm, "chat"):            
            try:
                outputs = self.llm.chat(messages=chat_inputs, 
                                        sampling_params=self._default_sampling_params if sampling_params is None else sampling_params)
                responses = [
                    o.outputs[0].text.strip() if o.outputs and len(o.outputs) > 0 else "<EMPTY>"
                    for o in outputs
                ]
            except Exception as e:
                print(f"[Warning] Chat generation failed: {e}, trying batch generate()...")

        if not responses:
            try:
                responses = self.generate(prompts, sampling_params)
            except Exception as e:
                print(f"[Warning] Batch generation failed in base class: {e}, retrying individually...")
                fallback_responses = []
                for idx, conv in enumerate(batch):
                    try:
                        prompt = conv.to_prompt(self)
                        response = self.generate([prompt], sampling_params)[0]
                    except Exception as e:
                        print(f"[Warn] Prompt #{idx} failed: {e}")
                        response = "<EMPTY>"
                    fallback_responses.append(response)
                responses = fallback_responses

        if len(responses) != len(batch):
            raise ValueError(f"Mismatch in batch size and response count. Batch size {len(batch)} != Response # {len(responses)}")
        
        self._total_response_count += len(responses)
        for idx, (conv, response) in enumerate(zip(batch, responses)):
            if isinstance(response, str):
                cleaned = clean_header_footer(response)
            else:
                cleaned = str(response).strip()

            normalized = cleaned.strip().lower()
            if not normalized or normalized == "no response" or normalized == "<empty>":
                
                try:
                    print(f"[Retry] Empty response for prompt #{idx+1}...")
                    prompt = conv.to_prompt(self)
                    retry_resp = self.generate([prompt], sampling_params)[0]
                    cleaned_retry = clean_header_footer(retry_resp) if isinstance(retry_resp, str) else str(retry_resp).strip()
                    
                    cleaned = cleaned_retry
                    response = retry_resp
                    responses[idx] = retry_resp
                    print(f"[Retry] New generation :{cleaned}")
                except Exception as e:
                    print(f"[Retry failed] Prompt #{idx+1}: {e}")

                retry_normalized = cleaned.strip().lower()
                if not retry_normalized or retry_normalized == "no response" or retry_normalized == "<empty>":
                    self._empty_response_count += 1
        
                    prompt = conv.get_last_message()
                    prompt_text = prompt.get("content", "") if prompt else ""
                    prompt_preview = " ".join(prompt_text.split())
                    if len(prompt_preview) > 160:
                        prompt_preview = f"{prompt_preview[:157]}..."
                    raw_response = response if isinstance(response, str) else str(response)
                    raw_preview = raw_response.strip() or "<empty>"
                    raw_preview = raw_preview if len(raw_preview) <= 80 else f"{raw_preview[:77]}..."
                    
                    error_msg = (f"Model {self.model_path} with generation config {self.generation_config} produced no usable output ({raw_preview})"
                            f"for prompt #{idx + 1}: {prompt_preview}, prompt length {len(prompt_text)}, Chat history: {chat_inputs[idx]}")
                    print(f"[Warning] {error_msg}")
                    cleaned = "No response"
                    responses[idx] = cleaned


            conv.add_message("assistant", cleaned)
        conv_messages = [conv.get_messages(self) for conv in batch]
        return conv_messages, responses

    def _apply_generation_config(self) -> None:
        cfg = self._generation_config
        default_cfg = dict(min_tokens=1, temperature=0.0, repetition_penalty=1.1, max_tokens=1024)

        try:
            if cfg:
                self._default_sampling_params = SamplingParams(**cfg) if cfg else SamplingParams()
            else:
                print(f"[Warning] Generation config not defined: {cfg}, using defaults.")
                self._default_sampling_params = SamplingParams(**default_cfg)
        except Exception as e:
            print(f"[Error] Failed to apply generation config: {cfg}, using defaults. Error: {e}")
            self._default_sampling_params = SamplingParams(**default_cfg)

    def normalize_batch(self, batch):
        """
        Normalize various input types into a list of conversation prompts.
        """
        if isinstance(batch, str):
            conv = Conversation()
            conv.add_message("user", batch)
            batch = [conv.to_prompt(self)]
        elif isinstance(batch, Conversation):
            batch = [batch.to_prompt(self)]
        elif isinstance(batch, list) and all(isinstance(b, Conversation) for b in batch):
            batch = [b.to_prompt(self) for b in batch]

        return batch
    
    def reset_empty_response_stats(self):
        self._empty_response_count = 0
        self._total_response_count = 0
        
    def check_empty_responses(self):
        if self._total_response_count == 0:
            return

        ratio = self._empty_response_count / self._total_response_count
        if ratio > self.EMPTY_RESPONSE_TOLERANCE:
            msg = (f"Model {self.model_path} exceeded empty response tolerance "
                f"({ratio:.2%} > {self.EMPTY_RESPONSE_TOLERANCE:.2%}). "
                f"Total={self._total_response_count}, Empty={self._empty_response_count}")
            if self.fail_on_empty_response:
                raise EmptyModelResponseError(msg)
            else:
                print(f"[Warning] {msg}")
    
    def _discover_supported_params(self, sampling_params):
        test_messages = [{"role": "user", "content": "Test message"}]
        params = {}
        
        for param_name in ["max_tokens", "max_output_tokens"]: # "max_completion_tokens"
            if self._test_param(test_messages, {param_name: sampling_params.max_tokens}):
                params[param_name] = sampling_params.max_tokens
                break
        
        param_tests = [
            ("temperature", sampling_params.temperature),
            ("top_p", sampling_params.top_p),
            ("top_k", sampling_params.top_k),
            ("presence_penalty", sampling_params.presence_penalty)
        ]
        
        for param_name, value in param_tests:
            test_params = params.copy()
            test_params[param_name] = value
            if self._test_param(test_messages, test_params):
                params[param_name] = value
        
        print(f"Discovered supported parameters: {list(params.keys())}")
        return params
        
    def _test_param(self, messages, params):
        raise NotImplementedError("This method must be implemented in subclasses.")
