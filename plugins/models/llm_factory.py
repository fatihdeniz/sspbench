from judge_models import JUDGE_MODELS
from llm_config import LLMConfig
from models import HuggingFaceLLM
from models import ApiLLM
from models import OpenaiLLM
from models import JudgeApiLLM
from models import AzureOpenaiLLM
from models import AzureInferenceLLM
from models import GoogleLLM

class LLMFactory:
    """
    A Factory class that creates LLM instance based on the given parameters
    """

    @staticmethod
    def create_llm(args):
        config = None
        try:
            if getattr(args, "judge", False):
                config = JUDGE_MODELS[args.test]
            elif getattr(args, "api_key", None) and getattr(args, "api_url", None):
                if args.model_type == "openai":
                    if "azure" in args.api_url:
                        config = LLMConfig(
                            type="azure",
                            model=args.model,
                            api_url=args.api_url,
                            api_token=args.api_key,
                            api_version=args.api_version if hasattr(args, "api_version") else None,
                        )
                    elif "google" in args.api_url:
                        config = LLMConfig(
                            type="google",
                            model=args.model,
                            api_url=args.api_url,
                            api_token=args.api_key,
                        )
                    else:
                        config = LLMConfig(
                            type="openai",
                            model=args.model,
                            api_url=args.api_url,
                            api_token=args.api_key,
                        )
                elif args.model_type == "api":
                    config = LLMConfig(
                        type="api",
                        model=args.model,
                        api_url=args.api_url,
                        api_token=args.api_key,
                    )
            else:
                config = LLMConfig(
                    type="huggingface",
                    model=args.model,
                )
        except Exception as e:
            print(f"Judge LLM config error: {repr(e)}")

        return LLMFactory.from_config(config) if config else None
    
    @staticmethod
    def from_config(config: LLMConfig, fail_on_empty_response=False):
        llm = None
        try:
            if config.type == "huggingface":
                llm = HuggingFaceLLM(config.model)
            elif config.type == "api":
                llm = JudgeApiLLM(config, fail_on_empty_response=fail_on_empty_response)
            elif config.type == "openai":
                if config.api_url and "azure" in config.api_url:
                    llm = AzureOpenaiLLM(config, fail_on_empty_response=fail_on_empty_response)
                else:
                    llm = OpenaiLLM(
                        base_url=config.api_url,
                        api_key=config.api_token,
                        model_name=config.model,
                        fail_on_empty_response=fail_on_empty_response,
                    )
            elif config.type == "azure":
                llm = OpenaiLLM(
                        base_url=config.api_url,
                        api_key=config.api_token,
                        model_name=config.model,
                        fail_on_empty_response=fail_on_empty_response,
                    )
            elif config.type == "google":
                llm = GoogleLLM(config)
        except Exception as e:
            print(f"LLM creation from config failed: {repr(e)}")

        if llm:
            print(
                f"Generated Model: {llm.__class__.__name__} "
                f"Fail on empty response: {getattr(llm, 'fail_on_empty_response', None)}"
            )
            if hasattr(llm, "model"):
                print(f"Model Name: {llm.model}")
            if hasattr(llm, "api_url"):
                print(f"API URL: {llm.api_url}")

        return llm