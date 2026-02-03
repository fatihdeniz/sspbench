import os
import torch
from typing import Optional

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

from vllm import LLM
from transformers import AutoTokenizer
from models.llm_base import LLMBase
from models.model_utils import get_attention_backend_order, temporary_env_var

DTYPE_QUANTIZATION_FALLBACKS = [
    (None, None),
    ("bfloat16", None),
    ("float16", None),
    ("float32", None),
    ("auto", "awq"),
    ("auto", "gptq"),
]

class HuggingFaceLLM(LLMBase):
    def __init__(self, model_path, 
                gpu_memory_utilization=0.9, 
                max_model_len=4096, 
                quantization=None,
                generation_config=None):
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = self._resolve_max_model_len(model_path, max_model_len)
        self.quantization = quantization
        self._role_support_cached = None
        self.apply_chat_template = True
        super().__init__(model_path, generation_config)

    def load_model(self):
        for backend in get_attention_backend_order(self.model_path):
            print(f"[Info] Trying attention backend: {backend or 'auto'}")
            for dtype, quantization in DTYPE_QUANTIZATION_FALLBACKS:
                llm = self._try_load_llm(dtype, quantization, backend)
                if llm is not None:
                    self.llm = llm
                    self.tokenizer = self.llm.get_tokenizer()
                    return
        
        raise RuntimeError(
            f"Failed to load LLM from {self.model_path}. "
            f"Strategies attempted: {DTYPE_QUANTIZATION_FALLBACKS}"
        )

    def generate(self, batch, sampling_params=None):
        batch = self.normalize_batch(batch)

        sampling_params = sampling_params or self._default_sampling_params

        responses = []
        try:
            batch_output = self.llm.generate(batch, sampling_params)
            responses = [response.outputs[0].text.strip() for response in batch_output]
        except Exception as e:
            print(f"[Warning] Batch generation failed for model {self.model_path}: {e}, trying individual generation...")
            for idx, prompt in enumerate(batch):
                try:
                    single_out = self.llm.generate(prompt, sampling_params)
                    text = single_out[0].outputs[0].text.strip()
                    if not text:
                        raise ValueError("Empty text")
                except Exception as e:
                    print(f"[Warn] Prompt #{idx} failed: {prompt!r} → {e}")
                    text = "No response"
                responses.append(text)
        
        return responses
    
    def check_role_support(self, role):
        if self._role_support_cached is not None:
            return self._role_support_cached
        
        conversation = [{"role": role, "content": "Test message"}]
        try:
            self.tokenizer.apply_chat_template(conversation, tokenize=False)
            self._role_support_cached = True
        except Exception as e:
            print(f"Failed to apply chat template with exception {e}")
            self._role_support_cached = False
        return self._role_support_cached

    def _resolve_max_model_len(self, model_path, max_model_len: int):
        if "meta-llama_Llama-Guard-4-12B" in model_path:
            return 8192
        try:
            tok = AutoTokenizer.from_pretrained(model_path, 
                                                trust_remote_code=self._resolve_trust_remote_code(model_path))
            tok_max = getattr(tok, "model_max_length", None)
            if tok_max and 0 < tok_max < 10_000_000:
                return min(int(tok_max), max_model_len)
        except Exception:
            pass

        return None
    
    def _resolve_trust_remote_code(self, model_path: str) -> bool:
        
        model_basepath = os.path.basename(model_path)
        owner = model_basepath.lower().split("_")[0]
        
        TRUSTED_OWNERS = {
            "openai", "google", "meta-llama", "facebook", "apple",
            "mistralai", "nvidia", "moonshotai", "deepseek-ai",
            "qcri", "falcon", "qwen", "microsoft", "xai-org",
        }

        if owner in TRUSTED_OWNERS:
            return True
        return False
    
    def _try_load_llm(
        self,
        dtype: Optional[str],
        quantization: Optional[str],
        attention_backend: Optional[str]
    ) -> Optional[LLM]:
        with temporary_env_var("VLLM_ATTENTION_BACKEND", attention_backend):
            gpu_count = 1 if quantization in ("awq", "gptq", "gguf", "bitsandbytes") else torch.cuda.device_count()
            kwargs = {
                "model": self.model_path,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": self._resolve_trust_remote_code(self.model_path),
                "tensor_parallel_size": gpu_count,
                "enforce_eager": True,
            }

            if self.max_model_len is not None:
                kwargs["max_model_len"] = int(self.max_model_len)

            if quantization:
                kwargs["quantization"] = quantization

            if dtype is not None:
                if kwargs.get("quantization") in ("awq", "gptq", "gguf"):
                    print(f"[Info] dtype={dtype} ignored for quant={kwargs['quantization']}")
                else:
                    kwargs["dtype"] = dtype
            else:
                kwargs["dtype"] = "auto"

            cfg = (
                f"dtype={kwargs.get('dtype', 'auto')} | "
                f"quant={kwargs.get('quantization', 'none')} | "
                f"backend={attention_backend or 'auto'} | "
                f"tp={kwargs['tensor_parallel_size']} | "
                f"max_len={kwargs.get('max_model_len', 'model-default')}"
            )
            print(f"[Try] Loading model with config: {cfg}")

            try:
                llm = LLM(**kwargs)
                if llm.generate("Hello, are you alive?")[0] == "No response":
                    raise RuntimeError("Model generated no response on test prompt.")
                print(f"[Success] Loaded with {cfg}")
                return llm

            except Exception as e:
                err_msg = str(e)
                if len(err_msg) > 350:
                    err_msg = err_msg[:350] + "..."

                print(
                    f"[Fail] {type(e).__name__} while loading model with {cfg}\n"
                    f"       Reason: {err_msg}"
                )
                return None
