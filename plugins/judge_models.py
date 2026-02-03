from collections import defaultdict
from config import Config
from llm_config import LLMConfig

def get_default_huggingface_model():
    return Config.JUDGE_HF_MODEL

def get_default_attacker_model():
    if Config.ATTACKER_MODEL_TYPE == "external":
        kwargs = {
            "type": "openai",
            "model": Config.ATTACKER_MODEL,
            "api_url": Config.ATTACKER_ENDPOINT,
            "api_token": Config.ATTACKER_TOKEN,
        }
        if getattr(Config, "ATTACKER_VERSION", None):
            kwargs["api_version"] = Config.ATTACKER_VERSION
        return LLMConfig(**kwargs)
        
def get_default_judge_model():
    if Config.JUDGE_MODEL_TYPE == "external":
        kwargs = {
            "type": "openai",
            "model": Config.JUDGE_EXTERNAL_MODEL,
            "api_url": Config.JUDGE_EXTERNAL_ENDPOINT,
            "api_token": Config.JUDGE_EXTERNAL_TOKEN,
        }
        if getattr(Config, "JUDGE_EXTERNAL_VERSION", None):
            kwargs["api_version"] = Config.JUDGE_EXTERNAL_VERSION
        return LLMConfig(**kwargs)
    elif Config.JUDGE_MODEL_TYPE == "internal":
        return LLMConfig(
            type="api",
            model=Config.JUDGE_INTERNAL_MODEL,
            api_url=Config.JUDGE_INTERNAL_ENDPOINT,
            api_token=Config.JUDGE_INTERNAL_TOKEN
        )
    return LLMConfig(
        type="huggingface",
        model=get_default_huggingface_model(),
        hf_token=Config.HF_TOKEN
    )

DEFAULT_JUDGE = get_default_judge_model()
ATTACKER_MODEL = get_default_attacker_model() or DEFAULT_JUDGE

VectaraConfig = LLMConfig(
    type="huggingface",
    model="vectara/hallucination_evaluation_model",
    hf_token=Config.HF_TOKEN
)

FlanT5Config = LLMConfig(
    type="huggingface",
    model="google/flan-t5-base",
    hf_token=Config.HF_TOKEN
)

JUDGE_MODELS = defaultdict(lambda: DEFAULT_JUDGE, {
    "simpleqa": DEFAULT_JUDGE,
    "selfcheckgpt": DEFAULT_JUDGE,
    "seccodeplt-instruct": ATTACKER_MODEL,
    "seccodeplt-autocomplete": ATTACKER_MODEL,
    "vectara": [FlanT5Config, VectaraConfig],
})
