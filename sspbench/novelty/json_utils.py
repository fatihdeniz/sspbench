"""
Utility functions for the Novelty Engine.
"""

import ast
from autogen.code_utils import extract_code


def extract_json_v2(json_text, outfilename):
    """
    Extract JSON from LLM response text.

    Args:
        json_text: The response text containing JSON
        outfilename: Optional filename to save the extracted JSON

    Returns:
        The extracted JSON object
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
        # load the json_string.
        json_dict = combined_json
        # json_dict = ast.literal_eval(combined_json)
        if outfilename is not None:
            import json
            with open(outfilename, "w") as f:
                json.dump(json_dict, f)

    else:
        assert False, "fail to output json file."
    return json_dict