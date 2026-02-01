import os
import sys
import nest_asyncio

nest_asyncio.apply()
parent_dir = os.path.dirname(os.getcwd())
base_dir = os.getcwd() + '/safety_llm'
sys.path.append(parent_dir)


async_list = ["gpt-4o"]


from trusteval.src.evaluation import lm_metric
base_dir='/Users/admin/Downloads/backup/trustgen-backup/LLM_ALL-1/safety'
lm_metric(
    base_dir=base_dir,
    aspect="safety_llm",
    model_list=async_list
)
# /Users/admin/Downloads/backup/trustgen-backup/LLM_ALL-1/safety/exaggerate_safety_responses_judge.json
