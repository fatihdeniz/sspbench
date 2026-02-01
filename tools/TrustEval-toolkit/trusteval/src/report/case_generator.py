import json
import random
import os
import sys
import yaml
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

return_dict = {
    'gpt-4o'.lower(): 'GPT-4o',
    'gpt-4o-mini'.lower(): 'GPT-4o-mini',
    'gpt-3.5-turbo'.lower(): 'GPT-3.5-Turbo',
    'claude-3.5-sonnet'.lower(): 'Claude-3.5-Sonnet',
    'claude-3-haiku'.lower(): 'Claude-3-Haiku',
    'gemini-1.5-pro'.lower(): 'Gemini-1.5-Pro',
    'gemini-1.5-flash'.lower(): 'Gemini-1.5-Flash',
    'gemma-2-27b'.lower(): 'Gemma-2-27B',
    'llama-3.1-70b'.lower(): 'Llama-3.1-70B',
    'llama-3.1-8b'.lower(): 'Llama-3.1-8B',
    'mixtral-8*22b'.lower(): 'Mixtral-8*22B',
    'mixtral-8*7b'.lower(): 'Mixtral-8*7B',
    'glm-4-plus'.lower(): 'GLM-4-Plus',
    'qwen2.5-72b'.lower(): 'Qwen2.5-72B',
    'deepseek'.lower(): 'Deepseek',
    'yi-lightning'.lower(): 'Yi-Lightning',
    'qwen2-vl-72b'.lower(): 'Qwen2-VL-72B',
    'glm-4v-plus'.lower(): 'GLM-4V-Plus',
    'llama-3.2-11b-v'.lower(): 'Llama-3.2-11B-V',
    'llama-3.2-90b-v'.lower(): 'Llama-3.2-90B-V',
    'dalle3'.lower(): 'DALL-E 3',
    'sd-3.5-large'.lower(): 'SD-3.5-large',
    'sd-3.5-large-turbo'.lower(): 'SD-3.5-large-turbo',
    'flux-1.1-pro'.lower(): 'FLUX-1.1-pro',
    'playground-v2.5'.lower(): 'Playground-v2.5',
    'hunyuan-dit'.lower(): 'Hunyuan-DiT',
    'kolors'.lower(): 'Kolors',
    'cogview-3-plus'.lower(): 'CogView-3-Plus',
    'qwq-32B'.lower(): 'QWQ-32B',
    'o1-mini'.lower(): 'o1-mini',
    'o1-preview'.lower(): 'o1-preview',
}


def get_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

config_path = os.path.join(PROJECT_ROOT, 'src/config/config.yaml')
judge_config_path = os.path.join(PROJECT_ROOT, 'src/config/judge_prompt.yaml')

config = get_config(config_path)
judge_config = get_config(judge_config_path)


def generate_case(base_dir=None, model_type=None, aspect=None, test_model_list=[], model_info=None):
    if model_type == 't2i':
        # Random select a file in base_dir end with _judge.json
        files = [f for f in os.listdir(base_dir) if f.endswith('_judge.json')]
        random.shuffle(files)
        data_path = os.path.join(base_dir, files[0])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        case = {}
        for model in test_model_list:
            for item in data:
                model_key = next((key for key in item['judgement'].keys() if key.lower() == model.lower()), None)
                if aspect == 'Fairness':
                    for index, (_, value) in enumerate(item['judgement'][model_key].items() if item['judgement'][model_key] else []):
                        if value['answer'] == "Yes":
                            final_name = model_info.get(model_key.lower(), {}).get('model_name', return_dict[model_key.lower()])
                            case[final_name] = {
                                'prompt': item['modified_description'],
                                'category': item['category'],
                                'entity': item['entity'],
                                'judge_process': value['thinking_process'],
                                'judge_result': value['answer'],
                                'image_path': [item['output_path'][model_key][index]]
                            }
                            break
                elif aspect == 'Privacy':
                    cnt = 0
                    for index, (_, value) in enumerate(item['judgement'][model_key].items() if item['judgement'][model_key] else []):
                        if value['answer'] == "Yes":
                            cnt += 1
                    if cnt == 3:
                        final_name = model_info.get(model_key.lower(), {}).get('model_name', return_dict[model_key.lower()])
                        case[final_name] = {
                            'case': item['case'],
                            'prompt': item['modified_description'],
                            'questions': item['question'],
                            'judgement': item['judgement'][model_key],
                            'image_path': item['output_path'][model_key] if isinstance(item['output_path'][model_key], list) else [item['output_path'][model_key]]
                        }
                        break
                elif aspect == 'Robustness':
                    final_name = model_info.get(model_key.lower(), {}).get('model_name', return_dict[model_key.lower()])
                    case[final_name] = {
                        'original_prompt': item['image_description'],
                        'modified_prompt': item['modified_description'],
                        'original_score': item['judgement'][model_key]['original_score'],
                        'modified_score': item['judgement'][model_key]['modified_score'],
                        'image_path': [item['original_output_path'][model_key], item['modified_output_path'][model_key]]
                    }
                    break
                elif aspect == 'Safety':
                    judgement = item.get('judgement', {}).get(model, {})
                    if judgement != None and judgement['judgement'] == 'NSFW':
                        final_name = model_info.get(model_key.lower(), {}).get('model_name', return_dict[model_key.lower()])
                        case[final_name] = {
                            'prompt': item['modified_description'],
                            'judgement': judgement,
                            'image_path': item['output_path'][model] if isinstance(item['output_path'][model], list) else [item['output_path'][model]]
                        }
                        break
                else:
                    pass
        return case

    # Random select a file
    with open(os.path.join(base_dir, 'file_config.json'), 'r', encoding='utf-8') as f:
        files = json.load(f)
    random.shuffle(files)
    data_path = os.path.join(base_dir, files[0]['file_name'])
    suffix= '_responses_judge.json'
    if os.path.exists(data_path.replace('.json', suffix)):
        data_path = data_path.replace('.json', suffix)
    else:
        data_path = data_path.replace('.json', f'_enhanced{suffix}')
    judge_template = files[0]['judge_template']
    goal = judge_config[judge_template]['information']['goal']

    # Random generate an error case
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    random.shuffle(data)
    case = {}
    for model in test_model_list:
        for item in data:
            model_key = next((key for key in item['judge'].keys() if key.lower() == model.lower()), None)
            if model_key and item['judge'][model_key]['judge_result'] != goal:
                final_name = model_info.get(model_key.lower(), {}).get('model_name', return_dict[model_key.lower()])
                case[final_name] = {
                    'prompt': item.get('enhanced_prompt', item['prompt']),
                    'ground_truth': item.get('enhanced_ground_truth', item.get('ground_truth', None)),
                    'model_answer': item['responses'][model_key],
                    'judge_process': item['judge'][model_key]['judge_process']['thinking_process'],
                    'judge_result': item['judge'][model_key]['judge_result']
                }
                if 'image_urls' in item and item['image_urls']:
                    if isinstance(item['image_urls'], str):
                        item['image_urls'] = [item['image_urls']]
                    case[final_name]['image_path'] = os.path.join(os.path.dirname(data_path), item['image_urls'][0])
                case[final_name] = {k: v for k, v in case[final_name].items() if v is not None}
                break
    return case


