import ast
import json
from autogen.code_utils import extract_code

try:
    import json5
except:
    pass


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


def safe_eval(s):
    if not s or not s.strip():
        raise ValueError("Empty string cannot be parsed")
    
    s = s.strip()
    try:
        return json5.loads(s)
    except Exception:
        pass
    
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    
    raise ValueError(f"Could not parse as JSON. First 200 chars: {s[:200]}")


def parse_json_response(response, fallback=None):
    if not response or not response.strip():
        return fallback
    try:
        return safe_eval(response)
    except Exception:
        pass
    
    try:
        extracted = extract_json_v2(response, None)
        if extracted:
            return extracted[0] if len(extracted) == 1 else extracted
    except Exception:
        pass
    
    return fallback


def extract_json_v2(json_text, outfilename):
    response = json_text.replace("TERMINATE", "")
    
    combined_json = []
    errors = []
    
    # Try to extract JSON code blocks
    try:
        if "```" in response:
            extracted_json = extract_code(response)
            
            # Try to parse each extracted code block
            for idx, (lang, code_block) in enumerate(extracted_json):
                try:
                    parsed = safe_eval(code_block)
                    # Ensure parsed result is a list
                    if isinstance(parsed, list):
                        combined_json.extend(parsed)
                    else:
                        combined_json.append(parsed)
                except Exception as e:
                    errors.append(f"Block {idx}: {str(e)[:150]}")
                    # Save problematic block for debugging
                    if outfilename:
                        error_file = outfilename.replace('.json', f'_error_block_{idx}.txt')
                        try:
                            with open(error_file, 'w') as f:
                                f.write(f"Error: {e}\n\n")
                                f.write(f"Code block:\n{code_block}\n")
                        except:
                            pass
    except Exception as e:
        errors.append(f"extract_code failed: {str(e)[:150]}")
    
    # Fallback: try to find JSON array pattern in text
    if not combined_json:
        import re
        json_pattern = r'\[\s*\{.*?\}\s*\]'
        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                parsed = safe_eval(match)
                if isinstance(parsed, list):
                    combined_json.extend(parsed)
                    break
            except:
                pass
    
    # If we couldn't parse any blocks, raise error with details
    if not combined_json:
        error_summary = "\n".join(errors) if errors else "No JSON found"
        raise ValueError(f"Failed to parse any JSON blocks.\n{error_summary}\nFirst 300 chars: {response[:300]}")
    
    json_dict = _flatten_to_dict_list(combined_json)
    
    if outfilename is not None:
        with open(outfilename, "w") as f:
            json.dump(json_dict, f, indent=2)
    
    return json_dict