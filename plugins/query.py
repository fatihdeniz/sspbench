import inspect
import pandas as pd
from typing import Any, Dict, Union
from models.llm_base import LLMBase
from conversation import Conversation

def get_field(row, field_name, default=""):
    """Safely get a field from a row with fallback."""
    return row.get(field_name, default)

def supports_parameter(func, param: str) -> bool:
    """Check if a function supports a given keyword argument."""
    return param in inspect.signature(func).parameters

def build_conversation(row, task_template, args, instruction, judge_field, prompt_field, response_field, target_field, target_type_field):
    """Construct the conversation based on task template and input row."""
    if task_template:
        if args.judge and judge_field in task_template:
            judge_data = task_template[judge_field]
            try:
                formatted_instruction = (
                    instruction.format(prompt=get_field(row, prompt_field))
                    if isinstance(instruction, str)
                    else instruction
                )
            except KeyError as e:
                formatted_instruction = instruction

            sys_prompt = get_field(judge_data, "system").format(
                instruction=formatted_instruction, prompt=get_field(row, prompt_field))
            conversation = Conversation(system_prompt=sys_prompt)
            
            user_prompt = get_field(judge_data, "user").format(
                prompt=get_field(row, prompt_field),
                target=get_field(row, target_field),
                target_type=get_field(row, target_type_field),
                response=get_field(row, response_field),
            )
            conversation.add_message("user", user_prompt)
        else:
            query_data = task_template["query"]
            sys_prompt = get_field(query_data, "system").format(
                instruction=instruction,
                target_type=get_field(row, target_type_field),
            )
            if "qwen" in args.model.lower():
                sys_prompt = "You are Qwen, created by Alibaba Cloud. " + sys_prompt   
            conversation = Conversation(system_prompt=sys_prompt)

            user_prompt = get_field(query_data, "user").format(
                prompt=get_field(row, prompt_field),
                target=get_field(row, target_field),
            )
            conversation.add_message("user", user_prompt)
    else:
        conversation = Conversation()
        conversation.add_message("user", get_field(row, prompt_field))

    return conversation

def generate_query(
    row: pd.Series,
    task_template: Union[Dict[str, Any], None],
    args: Any,
    llm: LLMBase,
    judge_field: str = "judge",
    instruction: Union[str, None] = None,
    prompt_field: str = "prompt",
    response_field: str = "response",
    target_field: str = "target",
    target_type_field: str = "category"
) -> Union[str, list]:
    """Generate a query string or structured conversation for LLM input."""
    
    conversation = build_conversation(
        row, task_template, args, instruction,
        judge_field, prompt_field, response_field, target_field, target_type_field)

    # return conversation.to_prompt(llm)
    return conversation
