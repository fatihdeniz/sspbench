import os
from typing import Optional
from contextlib import contextmanager

def get_attention_backend_order(model_path: str):
    m = model_path.lower()

    if "gemma-3" in m or "gemma3" in m:
        return [
            "FLASH_ATTN",
            "FLASHINFER",
            None,  
            "XFORMERS",
        ]
    if "gemma-2" in m or "gemma2" in m or "fanar" in m:
        return [
            "XFORMERS",
            None,            
            "FLASH_ATTN",
            "FLASHINFER",
        ]
    if "qwen" in m:
        return [
            "FLASHINFER",
            "FLASH_ATTN",
            None,
            "XFORMERS",
        ]
    if "llama" in m:
        return [
            "FLASH_ATTN",
            None,
            "XFORMERS",
        ]
    if "mistral" in m or "mixtral" in m:
        return [
            "FLASH_ATTN",
            None,
            "XFORMERS",
        ]
    return [
        "FLASH_ATTN",
        "FLASHINFER",
        None,
        "XFORMERS",
    ]
    
@contextmanager
def temporary_env_var(key: str, value: Optional[str]):
    original_value = os.environ.get(key)
    if value is not None:
        os.environ[key] = value
    elif key in os.environ:
        del os.environ[key]
    
    try:
        yield
    finally:
        if original_value is not None:
            os.environ[key] = original_value
        elif key in os.environ:
            del os.environ[key]