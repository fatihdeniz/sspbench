"""
Utility functions for the Novelty Engine.
"""

import ast
import json
from autogen.code_utils import extract_code


def _flatten_to_dict_list(obj):
    if isinstance(obj, dict):
        return [obj]
    elif isinstance(obj, list):
        result = []
        for item in obj:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, list):
                result.extend(_flatten_to_dict_list(item))
        return result
    return []


def extract_json_v2(json_text, outfilename):
    """
    Extract JSON from LLM response text.

    Args:
        json_text: The response text containing JSON
        outfilename: Optional filename to save the extracted JSON

    Returns:
        List of extracted JSON dictionaries
    """
    response = json_text.replace("TERMINATE", "")
    if "```json" in response:
        # parse the json file
        try:
            extracted_json = extract_code(response)
            combined_json = sum([], [ast.literal_eval(xx[1]) for xx in extracted_json])
        except:
            if '...' in response:
                response = response.replace('...', '')
                extracted_json = extract_code(response)
                combined_json = sum([], [ast.literal_eval(xx[1]) for xx in extracted_json])
            else:
                response2 = "\n".join(response.split('\n')[:-1]) + "]\n```"
                extracted_json = extract_code(response2)
                combined_json = sum([], [ast.literal_eval(xx[1]) for xx in extracted_json])
        
        json_dict = _flatten_to_dict_list(combined_json)
        
        if outfilename is not None:
            with open(outfilename, "w") as f:
                json.dump(json_dict, f, indent=2)

    else:
        assert False, "fail to output json file."
    return json_dict