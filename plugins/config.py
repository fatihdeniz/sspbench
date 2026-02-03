import os
from pathlib import Path
from dotenv import load_dotenv

# Find and load .env file
# This searches for .env starting from the current file's directory up to the project root
env_path = Path(__file__).resolve()
for parent in [env_path.parent] + list(env_path.parents):
    env_file = parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        break

class Config:
    JUDGE_MODEL_TYPE = os.getenv("JUDGE_MODEL_TYPE")
    HF_TOKEN = os.getenv("HF_TOKEN")
    JUDGE_HF_MODEL = os.getenv("JUDGE_HF_MODEL")
    JUDGE_INTERNAL_ENDPOINT = os.getenv("JUDGE_INTERNAL_ENDPOINT")
    JUDGE_INTERNAL_MODEL = os.getenv("JUDGE_INTERNAL_MODEL")
    JUDGE_INTERNAL_TOKEN = os.getenv("JUDGE_INTERNAL_TOKEN")
    JUDGE_EXTERNAL_ENDPOINT = os.getenv("JUDGE_EXTERNAL_ENDPOINT")
    JUDGE_EXTERNAL_MODEL = os.getenv("JUDGE_EXTERNAL_MODEL")
    JUDGE_EXTERNAL_TOKEN = os.getenv("JUDGE_EXTERNAL_TOKEN")
    JUDGE_EXTERNAL_VERSION = os.getenv("JUDGE_EXTERNAL_VERSION")
    
    ATTACKER_MODEL_TYPE = os.getenv("ATTACKER_MODEL_TYPE")
    ATTACKER_ENDPOINT = os.getenv("ATTACKER_ENDPOINT")
    ATTACKER_MODEL = os.getenv("ATTACKER_MODEL")
    ATTACKER_TOKEN = os.getenv("ATTACKER_TOKEN")
    ATTACKER_VERSION = os.getenv("ATTACKER_VERSION")
# import os

# class Config:
#     JUDGE_MODEL_TYPE = os.getenv("JUDGE_MODEL_TYPE")
#     HF_TOKEN = os.getenv("HF_TOKEN")
#     JUDGE_HF_MODEL = os.getenv("JUDGE_HF_MODEL")
#     JUDGE_INTERNAL_ENDPOINT = os.getenv("JUDGE_INTERNAL_ENDPOINT")
#     JUDGE_INTERNAL_MODEL = os.getenv("JUDGE_INTERNAL_MODEL")
#     JUDGE_INTERNAL_TOKEN = os.getenv("JUDGE_INTERNAL_TOKEN")
#     JUDGE_EXTERNAL_ENDPOINT = os.getenv("JUDGE_EXTERNAL_ENDPOINT")
#     JUDGE_EXTERNAL_MODEL = os.getenv("JUDGE_EXTERNAL_MODEL")
#     JUDGE_EXTERNAL_TOKEN = os.getenv("JUDGE_EXTERNAL_TOKEN")
#     JUDGE_EXTERNAL_VERSION = os.getenv("JUDGE_EXTERNAL_VERSION")
    
#     ATTACKER_MODEL_TYPE = os.getenv("ATTACKER_MODEL_TYPE")
#     ATTACKER_ENDPOINT = os.getenv("ATTACKER_ENDPOINT")
#     ATTACKER_MODEL = os.getenv("ATTACKER_MODEL")
#     ATTACKER_TOKEN = os.getenv("ATTACKER_TOKEN")
#     ATTACKER_VERSION = os.getenv("ATTACKER_VERSION")