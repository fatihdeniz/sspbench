import os
import sys
import nest_asyncio
nest_asyncio.apply()
import asyncio
## conmment this line if you don't need to use proxy
os.environ['http_proxy'] = 'http://127.0.0.1:7890' 
os.environ['https_proxy'] = 'http://127.0.0.1:7890'  

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

base_dir = os.path.abspath("./prompt_injection")
from trusteval.src.evaluation import judge_responses
from trusteval.src.response_generator.lm_response import generate_responses
from trusteval.src.evaluation import lm_metric

import shutil

async def main():
    data_folder  = base_dir
    model_list = ['gpt-4o']
    async_judge_model = ['gpt-4o-mini']
    response_key = ['responses']
    judge_key = 'judge'
    judge_type='llm'
    lm_metric(
        base_dir=data_folder,
        aspect="prompt_injection_llm",
        model_list=model_list
    )
if __name__ == "__main__":
    asyncio.run(main())
