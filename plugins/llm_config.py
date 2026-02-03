from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    type: str
    model: str
    api_url: Optional[str] = None
    api_token: Optional[str] = None
    api_version: Optional[str] = None
    hf_token: Optional[str] = None