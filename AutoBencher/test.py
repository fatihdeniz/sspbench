import re, time, os, argparse, ast, json, tqdm
import glob
import random
import copy
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from time import sleep
from collections import defaultdict
import numpy as np
# from util import gen_from_prompt, load_model, process_args_for_models, helm_process_args
from tool_util import _generate_lm_answers, extract_json_v2
# from tool_util import search_related_pages, search_step, get_pageviews

import requests
from bs4 import BeautifulSoup



DEFAULT_JSON_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your reasoning and language skills.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
Reply "TERMINATE" in the end when everything is done.
"""


def get_pageviews(page_title, start_date="2020040100", end_date="2026010100"):
    access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIwMDFkMTFmNmQ2MzVmMGY4YmI3MDlkNWViN2ZhNDRlYiIsImp0aSI6IjMzOGQ0Mzc0YzNmZjE5NjBlZDkzNjIwNTdiYjMwYjExOWYzZTY2MzVkZjM3NmY3NDcyZjczMDcyMjNiYzU4ODFjODBkOTliOTZmMjAzZGNkIiwiaWF0IjoxNzEyNjEwMTg0LjY4OTIyNywibmJmIjoxNzEyNjEwMTg0LjY4OTIzLCJleHAiOjMzMjY5NTE4OTg0LjY4NzY1Mywic3ViIjoiNzUzODczODIiLCJpc3MiOiJodHRwczovL21ldGEud2lraW1lZGlhLm9yZyIsInJhdGVsaW1pdCI6eyJyZXF1ZXN0c19wZXJfdW5pdCI6NTAwMCwidW5pdCI6IkhPVVIifSwic2NvcGVzIjpbImJhc2ljIl19.YN0ZvSzsBuYe3Mg-r0C63cWxDXPU3GOCyspUqg4mMv27Qw1FJq9F9H6JKJAUMrqQxB-xyWZqpu8mekvMoxb3Ha5S2fpPbuM4gMB0JketqG2obaDd4QqgtJjg8KDYKwR8ieKoPRLDSHv3Tv4NcvIL-EvzjkRybqrukzQwttwuBUwxmlY8vhC1BZed7URt_-KhMYPsnNfJLSBeWivYJOmrqF2S04AOS0Egjul8Pz_yXAQ7q7aqpIwg6X2jod0ZN5h1gnmAvZmoLB7mKSAxrHEUL2zaQ8BVERWostWVA9ek556cuUJe5NusQ0XW7pcsYIi0YpFjKOBuq-tXzuOlbxFhlbwrp6xkhE_grQGNs1IxyT-w_sjQc2gI48FDe0ldDrTg6ZmgLELsjJM8xOxBy1ng1fY73p-QnaDdxX4hqRw2ZBDlZ1E2j84lvVrv62x_SHPiBNAeywEPcOqDRV_XbU6ArOyJ7QTZXRu9UOT0XDQ-Fx3maCRGb35W4aOtLSWL-SSXYLI8ZuOQ2BwKQQYYbEDMp0W7NjHWzh8YPv6Y2wDaMzsAqaxk2c36pNvTToiTc_P6_a56lydQwoT8ACx1kzzw5lTNPKPEPxPGNiMgtsL3VqtxJWMR7Lgq-ZKwI7cwQ5FTp2YriQDBYuvoaDQeG_eVh8BlNlyg26OYojtYbNos3os"
    client_id = "001d11f6d635f0f8bb709d5eb7fa44eb"
    client_secret = "630b434daa4c8f6cce03b1c294b59574c1ce9431"  # Example client secret
    headers = {
        'Authorization': f'Bearer {access_token}',
        'User-Agent': 'wikipagerank',
    }

    # Construct the API URL with the appropriate parameters
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{page_title}/daily/{start_date}/{end_date}"
    # Make the HTTP GET request to the API
    response = requests.get(url, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Extract the pageview data
        print('retrieved for ', page_title)
        views = sum(item['views'] for item in data['items'])
        return views
    else:
        print(f"Failed to retrieve pageviews data for {page_title}. Status code: {response.status_code}")
        return 0
    
def clean_str(p):
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except:
        return ''

def filter_paragraph(paragraph_lst):
    return [p for p in paragraph_lst if len(p.split(" ")) > 2 and len(p.split(".")) > 1]

def get_page_obs(page):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs

def search_step(entity, output_more=False):
    headers = {
        "User-Agent": "autobencher-data-curation/1.0 (research)"
    }

    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    response_text = requests.get(search_url, headers=headers).text
    soup = BeautifulSoup(response_text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    if result_divs:  # mismatch
        result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
        # obs = f"Could not find {entity}. Similar: {result_titles[:5]}."
        print(f"Could not find {entity}. Search for similar entities, {result_titles[0]}, instead")
        # obs, entity = search_step(result_titles[0])
        obs, entity, wiki_url = search_step(result_titles[0])
    else:
        print('found entity', entity)
        canonical = soup.find("link", rel="canonical")
        wiki_url = canonical["href"] if canonical else response.url
        
        page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        if any("may refer to:" in p for p in page):
            # obs, entity = search_step("[" + entity + "]")
            obs, entity, wiki_url = search_step("[" + entity + "]")
        else:
            page_ = ""
            for p in page:
                if len(p.split(" ")) > 2:
                    page_ += clean_str(p)
                    if not p.endswith("\n"):
                        page_ += "\n"
            obs = get_page_obs(page_)
            if output_more:
                obs = filter_paragraph(obs)
            else:
                obs = filter_paragraph(obs[:10])

    return obs, entity, wiki_url

def gen_from_prompt(model, prompt, temperature=0., max_tokens=20):
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    
    is_single = isinstance(prompt, str)
    prompts = [prompt] if is_single else prompt

    convs = []
    for p in prompts:
        conv = Conversation()
        conv.add_message("user", p)
        convs.append(conv)

    responses = model.generate(convs, sampling_params)
    # print("Conversation", convs[0].to_list(), "Response:", responses[0])
    return responses[0] if is_single else responses

def search_related_pages(search_query):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": search_query,
        "srlimit": "max",
        "format": "json"
    }
    headers = {
        "User-Agent": "autobencher-data-curation/1.0 (research)"
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        print("Wikipedia API HTTP error:", response.status_code)
        print(response.text)
        return []

    data = response.json()
    if "query" not in data or "search" not in data["query"]:
        print("Wikipedia API returned unexpected payload:")
        print(data)
        return []

    return [item["title"] for item in data["query"]["search"]]

def test_taker_inference(test_model_info, problem_json, outfile, bsz=1, temperature=0.01, max_length=50):
    # if len(test_model_info) == 3:
    #     model_choice, tokenizer_choice, client_choice = test_model_info
    #     auth = None
    #     use_helm = False
    # elif len(test_model_info) == 4:
    #     model_choice, tokenizer_choice, client_choice, auth = test_model_info
    #     use_helm = True

    print(f'writing to {outfile}')
    out_handle = open(outfile, 'w')
    full_result_lst = []
    batch_lst, line_lst = [], []
    for line in tqdm.tqdm(problem_json):
        line['prompt'] = "Output just with the final answer to the question.\nQuestion:" + line[
            'question'] + "\n" + "Answer:"
        line_lst.append(line)
        batch_lst.append(line['prompt'])
        if len(batch_lst) < bsz:
            continue  # batch not full yet
        responses = gen_from_prompt(model=test_model_info, prompt=batch_lst, temperature=temperature, max_tokens=max_length)

        for line, xx in zip(line_lst, responses):
            # print(line['prompt'])
            # print('-' * 100)
            # print(xx.text)
            line['test_taker_response'] = xx
            print(json.dumps(line), file=out_handle)
            full_result_lst.append(line)
        batch_lst, line_lst = [], []
    if len(batch_lst) > 0:
        responses = gen_from_prompt(model=test_model_info, prompt=batch_lst, temperature=temperature, max_tokens=max_length)
        for line, xx in zip(line_lst, responses):
            line['test_taker_response'] = xx
            print(json.dumps(line), file=out_handle)
            full_result_lst.append(line)
    out_handle.close()
    return full_result_lst


def _generate_lm_answers(question_inputs, test_model_info, outfile_prefix='att1'):
    if os.path.exists(f"{outfile_prefix}.test_taker_inference.json"):
        full_result_lst = []
        with open(f"{outfile_prefix}.test_taker_inference.json", 'r') as in_handle:
            for line in in_handle:
                line = json.loads(line.strip())
                full_result_lst.append(line)
        return full_result_lst

    # test_taker_lm, test_taker_tokenizer, test_taker_client = test_model_info
    if isinstance(question_inputs, list) or isinstance(question_inputs, dict):
        question_inputs_str = json.dumps(question_inputs, indent=2)
    else:
        assert False

    if isinstance(question_inputs, list) and isinstance(question_inputs[0], list):
        json_dict = question_inputs[0]
    elif isinstance(question_inputs, list):
        json_dict = question_inputs
    else:
        print('question_inputs should be a list.')
        assert False

    full_result_lst = test_taker_inference(test_model_info, json_dict,
                                           outfile=f"{outfile_prefix}.test_taker_inference.json")

    return full_result_lst

def get_summary_of_results(json_dict, gold_key="python_answer", verbose=False):
    # a summary of the results.
    # summarize by each category.
    category2correct_count = defaultdict(list)
    category2question = defaultdict(list)
    str_summary = 'In the following, we summarize the evaluation results by each category in this agent iteration. \n We will report the accuracy for each category, and list the questions that are answered correctly and incorrectly. \n'
    for line in json_dict:
        line['category2'] = f"{line['category']} || {line['wiki_entity']} [{line['additional_requirement']}]" if 'additional_requirement' in line else line['category']
        category2correct_count[line['category2']].append(line['is_correct'])
        category2question[(line['category2'], line['is_correct'])].append(line)
    for category in category2correct_count:
        acc_temp = sum([1 if x == 'true' else 0 for x in category2correct_count[category]]) / len(category2correct_count[category])
        str_summary += f"category: {category}, accuracy: {round(acc_temp, 3)} " \
                       f"|| {sum([1 if x == 'true' else 0 for x in category2correct_count[category]])} out of {len(category2correct_count[category])}" + "\n"
        if verbose:
            str_summary += "# Questions answered correctly:\n"
            for qq in category2question[(category, 'true')]:
                str_summary += f"{qq['question']} || gold: {qq[gold_key]} || pred: {qq['test_taker_answer']}" + "\n"

                # str_summary += f"{qq['question']} || {qq['difficulty']} || gold: {qq['python_answer']} || pred: {qq['test_taker_answer']}" + "\n"
            str_summary += "# Questions answered incorrectly:\n"
            for qq in category2question[(category, 'false')]:
                str_summary += f"{qq['question']} || gold: {qq[gold_key]} || pred: {qq['test_taker_answer']}" + "\n"
            str_summary += "\n + ------------------------------------ + \n"
    # print(str_summary)
    return str_summary

def summarize_over_history(history_json_dict, gold_key="python_answer", verbose=True):
    '''
    :param history: a list of dictionaries. Each dictionary corresponds to a run.
    :return: a summary of the results.
    '''
    # augment each line of the dictionary with the iteration number.
    for idx, json_dict in enumerate(history_json_dict):
        for line in json_dict:
            line['iteration'] = idx
    # concatenate the dictionaries.
    json_dict = [line for json_dict in history_json_dict for line in json_dict]
    # a summary of the results.
    str_summary = get_summary_of_results(json_dict, gold_key=gold_key, verbose=verbose)
    # print(str_summary)
    return str_summary


def _refine_categories_random_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
    category_json = _generate_categories_random_augmented(theme, agent_info, history, iters, outfile_prefix=outfile_prefix+'.brainstorm', acc_target=acc_target)
    # given the json_lst, refine the categories to achieve the target accuracy.
    full_cat_lst = []
    for line in category_json:
        cat_lst = search_related_pages(line['category'])
        full_cat_lst.extend(cat_lst)
    context = """ Your goal is to select from a list of categories for knowledge intensive questions so that the selected subset are not repetitive from prior selectioins and covers a wide range of topics that are important.
The categories should be selected based on three criteria: (1) aligned with THEME, (2) salient and cover important topics.
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.

Output Formatting: 
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement. 
Make sure the categories are similar to wikipedia categories. 
The categories should be exactly in the following format (a list of dictionaries): 
```json 
[
{"id": "1", "category": "Ancient Philosophers", "parent_category": "History", "additional_requirement": "only ask about famous people and their ideologies"}, 
{"id": "2", "category": "Second World War", "parent_category": "History", "additional_requirement": "major battles"}, 
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration: 
The goal is to find a set of categories that have broad coverage of topics and are not repetitive from prior selections. 

At every iteration, you are given a list of categories that you have already explored and their respective accuracy. Also, you are given a larger set of candidate categories for this iteration, and you should use the information from previous iterations to select the top 10 categories from the list. 
DO NOT REPEAT any of the categories that you have already explored.
"""
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _refine_categories(theme, context, agent_info, history, iters, full_cat_lst, outfile_prefix=outfile_prefix + '.refine')

def _refine_categories_targetacc_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
    category_json = _generate_categories_targetacc_augmented(theme, agent_info, history, iters, outfile_prefix=outfile_prefix+'.brainstorm', acc_target=acc_target)
    # given the json_lst, refine the categories to achieve the target accuracy.
    full_cat_lst = []
    for line in category_json:
        cat_lst = search_related_pages(line['category'])
        full_cat_lst.extend(cat_lst)
    context = """ Your goal is to select from a list of categories for knowledge intensive questions so that the selected subset are likely to achieve the target accuracy of {ACC_TARGET}.
The categories should be selected based on three criteria: (1) aligned with THEME, (2) likely to obtain the target accuracy of {ACC_TARGET}, you can judge this based on the accuracy statistics from previous iterations. and (3) salient and cover important topics.
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.

Output Formatting: 
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement. 
Make sure the categories are similar to wikipedia categories. 
The categories should be exactly in the following format (a list of dictionaries): 
```json 
[
{"id": "1", "category": "Ancient Philosophers", "parent_category": "History", "additional_requirement": "only ask about famous people and their ideologies"}, 
{"id": "2", "category": "Second World War", "parent_category": "History", "additional_requirement": "major battles"}, 
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration: 
The goal is to find a set of categories that with accuracy close to the target accuracy level of {ACC_TARGET}. 

At every iteration, you are given a list of categories that you have already explored and their respective accuracy. Also, you are given a larger set of candidate categories for this iteration, and you should use the information from previous iterations to select the top 10 categories from the list, that are most likely to achieve the target accuracy level, while still being relevant and salient. 
In later iterations you should receive as input the categories that you have already explored and their respective accuracy. You should
DO NOT REPEAT any of the categories that you have already explored.
"""
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _refine_categories(theme, context, agent_info, history, iters, full_cat_lst, outfile_prefix=outfile_prefix + '.refine')




def _generate_categories_targetacc_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
    context = """ Your goal is to come up with a list of categories for knowledge intensive questions that achieve the target accuracy of {ACC_TARGET}.
The categories should be diverse and cover important topics, under the theme of THEME. 
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.
Constructing the categories is like building a tree structure of history, and (category, parent_category) is like specifying a node and its parent. We should select the most precise parent category, for example if you are trying to expand the category "second world war" to make it more specific by adding the node "famous battles in second world war", you should specify the parent category as "second world war" instead of "history".

Output Formatting: 
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement. 
Make sure the categories are similar to wikipedia categories. 
The categories should be exactly in the following format (a list of dictionaries): 
```json 
[
{"id": "1", "category": "Ancient Philosophers", "parent_category": "History", "additional_requirement": "only ask about famous people and their ideologies"}, 
{"id": "2", "category": "Second World War", "parent_category": "History", "additional_requirement": "major battles"}, 
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration: 
The goal is to find a set of categories that with accuracy close to the target accuracy level of {ACC_TARGET}. 

For iteration 1, you can start with a wide variety of categories for us to build upon later. 
In later iterations you should receive as input the categories that you have already explored and their respective accuracy. You should
1. Think about breadth. Brainstorm questions with different categories to have broader coverage. Coming up with new categories that can are likely to achieve the target accuracy level.
2. For example, If you find the model now lacks categories of 0.3 -- 0.5 accuracy, you should come up with more categories that would yield accuracy in that range, by either reducing the difficulty of questions that achieve lower accuracy (via subcategory or via additional requirement), or increasing the difficulty of questions that achieve higher accuracy.
3. DO NOT REPEAT any of the categories that you have already explored.
"""
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _generate_categories(theme, context, agent_info, history, iters, outfile_prefix=outfile_prefix)

def _generate_categories_random_augmented(theme, agent_info, history, iters, outfile_prefix='att1', acc_target="0.3--0.5"):
    context = """ Your goal is to come up with a list of categories for knowledge intensive questions that have broad coverage and are salient. 
The categories should be diverse and cover important topics, under the theme of THEME. 
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.
Constructing the categories is like building a tree structure of history, and (category, parent_category) is like specifying a node and its parent. We should select the most precise parent category, for example if you are trying to expand the category "second world war" to make it more specific by adding the node "famous battles in second world war", you should specify the parent category as "second world war" instead of "history".

Output Formatting: 
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement. 
Make sure the categories are similar to wikipedia categories. 
The categories should be exactly in the following format (a list of dictionaries): 
```json 
[
{"id": "1", "category": "Ancient Philosophers", "parent_category": "History", "additional_requirement": "only ask about famous people and their ideologies"}, 
{"id": "2", "category": "Second World War", "parent_category": "History", "additional_requirement": "major battles"}, 
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration: 
The goal is to find a set of categories that have broad coverage of topics and are salient. 

For iteration 1, you can start with a wide variety of categories for us to build upon later. 
In later iterations you should receive as input the categories that you have already explored and their respective accuracy. You should
1. Think about breadth. Brainstorm questions with different categories to have broader coverage.
2. DO NOT REPEAT any of the categories that you have already explored.
"""
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _generate_categories(theme, context, agent_info, history, iters, outfile_prefix=outfile_prefix)

def _refine_categories(theme, context, agent_info, history, iters, candidate_lst, outfile_prefix='att1'):
    if os.path.exists(f"{outfile_prefix}.categories.json"):
        print("FOUND categories.json")
        return json.load(open(f"{outfile_prefix}.categories.json", "r"))[0]
    # agent_lm, agent_tokenizer, agent_client = agent_info
    context = context.replace("THEME", theme)
    if iters is None:
        iters = len(history) + 1
    if iters == 1:
        context += "Please start with iteration 1." + "Here are the category candidates to select from (delimited by ||): " + " || ".join(candidate_lst) + "\n"
    else:
        context += "\n".join(history) + "Please start with iteration {}.".format(iters) + "Here are the category candidates to select from (delimited by ||): " + "||".join(candidate_lst) + "\n"
    context = DEFAULT_JSON_MESSAGE + context
    # # extract the json file from the message
    # request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
    #                                  echo_prompt=False, temperature=0.0, max_tokens=2000,
    #                                  process_func=None, service=agent_client,
    #                                  terminate_by_linebreak='no',)
    # response = request_result.completions[0].text
    response = gen_from_prompt(model=agent_info, prompt=context, temperature=0.0, max_tokens=2000)


    with open(f"{outfile_prefix}.full_thoughts.txt", 'w', encoding='utf-8') as out_handle:
        out_handle.write(context)
        out_handle.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        out_handle.write(response)

    extracted_json = extract_json_v2(response, f"{outfile_prefix}.categories.json")
    if len(extracted_json) == 1:
        extracted_json = extracted_json[0]
    return extracted_json

def _generate_categories(theme, context, agent_info, history, iters, outfile_prefix='att1'):
    if os.path.exists(f"{outfile_prefix}.categories.json"):
        print("FOUND categories.json")
        return json.load(open(f"{outfile_prefix}.categories.json", "r"))[0]
    # agent_lm, agent_tokenizer, agent_client = agent_info
    context = context.replace("THEME", theme)
    if iters is None:
        iters = len(history) + 1
    if iters == 1:
        context += "Please start with iteration 1."
    else:
        context += "\n".join(history) + "Please start with iteration {}.".format(iters)
    context = DEFAULT_JSON_MESSAGE + context
    # extract the json file from the message
    
    # request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
    #                                  echo_prompt=False, temperature=0.0, max_tokens=2000,
    #                                  process_func=None, service=agent_client,
    #                                  terminate_by_linebreak='no', )
    # response = request_result.completions[0].text
    
    response = gen_from_prompt(model=agent_info, prompt=context, temperature=0.0, max_tokens=2000)

    with open(f"{outfile_prefix}.full_thoughts.txt", 'w', encoding='utf-8') as out_handle:
        out_handle.write(context)
        out_handle.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        out_handle.write(response)

    extracted_json = extract_json_v2(response, f"{outfile_prefix}.categories.json")
    if len(extracted_json) == 1:
        extracted_json = extracted_json[0]
    return extracted_json






def get_acc_lst(json_dict, gold_key="python_answer"):
    # a summary of the results.
    # summarize by each category.
    category2correct_count = defaultdict(list)
    for line in json_dict:
        category2correct_count[line['category']].append(line['is_correct'])
    acc_lst = []
    for category in category2correct_count:
        acc = sum([1 if x == 'true' else 0 for x in category2correct_count[category]]) / len(category2correct_count[category])
        acc_lst.append(acc)
    return acc_lst



def solve_and_compare_questions(test_taker_info, agent_info, question_json, gold_answer, outfile_prefix, gold_ans_key='gold_answer'):
    test_taker_output = _generate_lm_answers(question_json,
                         test_taker_info,
                         outfile_prefix=outfile_prefix)
    summary_prev_iteration, history_json = fast_compare_answers(gold_answer, test_taker_output,
                                                                agent_info, outfile_prefix=outfile_prefix,
                                                                gold_ans_key=gold_ans_key)

    return history_json


def fast_compare_answers(gold_output, test_taker_output, agent_model_info, outfile_prefix='att1', gold_ans_key='gold_answer'):
    if os.path.exists(f"{outfile_prefix}.compare_answers.json"):
        print('FOUND compare_answers.json')
        json_dict = json.load(open(f"{outfile_prefix}.compare_answers.json", "r"))
        str_summary = get_summary_of_results(json_dict, gold_key="gold_answer")
        return str_summary, json_dict

    print("Comparing the answers generated by the python code and the test taker...")
    # agent_lm, agent_tokenizer, agent_client = agent_model_info
    print(len(gold_output), len(test_taker_output))
    assert len(gold_output) == len(test_taker_output)
    context_str = """Your goal is to compare the prediction with the gold answer, and judge the correctness of the prediction. 
We'd still consider the prediction to be correct if 
1. the prediction is semantically the same as the gold answer: formating or different way of reference shouldn't affect correctness. For example, if the gold answer is Jan 21, and the test taker output is 01/21, we would still consider the prediction to be correct. For example, United States and USA refer to the same entity.
2. the prediction refers a broader entity that contains the gold answer. For example, if the gold answer is Beijing, and the test taker output is Asia, we will then consider correctness based on the question.
3. If the question is slightly ambiguous, such that there are multiple correct answers: For example, if the question asks for reasons why something happens, and it could be caused by multiple reasons, we will consider the prediction to be correct if the prediction contains one of the correct answers.

You should output a short and succinct reasoning for the your correctness prediction. Then, you should output delimiter "##" and output "true" if the prediction is correct, and "false" if the prediction is incorrect.
Example Format: 
Question: What is 1+1?
pred=2 || gold=2.0
reason: identical numbers ## true
"""
    out_handle = open(f"{outfile_prefix}.compare_answers.jsonl", 'w')
    final_lst = []
    correct_count2 = 0
    for idx, (line_gold, line_pred) in tqdm.tqdm(enumerate(zip(gold_output, test_taker_output))):
        # print(line_gold, line_pred)
        line = {'id': str(idx + 1), 'question': line_gold['question'], 'gold_answer': line_gold[gold_ans_key],
                "test_taker_answer": line_pred['test_taker_response']}
        # add other fields in line_gold to line.
        for k, v in line_gold.items():
            if k not in line:
                line[k] = v
        pred = line_pred['test_taker_response'].strip()
        gold = line_gold[gold_ans_key].strip()
        q_str = f"Question {idx+1}: {line_gold['question']}\npred={pred} || gold={gold}\nreason:"
        context = context_str + q_str
        # request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
        #                                  echo_prompt=False, temperature=0.0, max_tokens=3000,
        #                                  process_func=None, service=agent_client,
        #                                  terminate_by_linebreak='no', verbose=False)
        # response = request_result.completions[0].text
        response = gen_from_prompt(model=agent_model_info, prompt=context, temperature=0.0, max_tokens=3000)
        
        
        line['reasons'] = response.strip()
        line['is_correct'] = response.strip().split('##')[-1].strip()
        test_taker_line = test_taker_output[idx]
        line['question'] = test_taker_line['question']
        if 'category' in test_taker_line:
            line['category'] = test_taker_line['category']
        else:
            line['category'] = 'None'
        if 'difficulty' in test_taker_line:
            line['difficulty'] = test_taker_line['difficulty']
        if line["is_correct"] == 'true':
            correct_count2 += 1

        print(json.dumps(line), file=out_handle)

        final_lst.append(line)
    json_dict = final_lst
    accuracy = correct_count2 / len(json_dict)
    print("accuracy: ", accuracy)
    assert len(json_dict) == len(test_taker_output)
    out_handle.close()


    with open(f"{outfile_prefix}.compare_answers.json", 'w') as out_handle:
        json.dump(json_dict, out_handle, indent=2)

    str_summary = get_summary_of_results(json_dict, gold_key="gold_answer")
    return str_summary, json_dict

def gen_qa_without_docs(topic, agent_info, additional_req):
    context = """You will generate a few question and answer pairs on the topic: {{TOPIC}}
Make sure not to ask subjective questions, and let the question's correct answer be a concise short phrase. 
Make the answer concise. 

You will also receive additional requirements on the questions. You should follow these additional requirements when generating the questions.
For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.

Try to generate a diverse set of 50 questions, and make sure that the questions are not too similar to each other while satisfying the additional requirements. If you can't generate 50 questions, generate as many as you can.

Formatting: 
Each question should be a dictionary with the following keys: id, question, answer, estimated difficulty. 
The questions should be exactly in the following format (a list of dictionaries): 
```json
{"id": "1", "question": "<question>", "answer": "<answer>", "difficulty": "1"}, 
{"id": "2", "question": "<question>", "answer": "<answer>", "difficulty": "1"}, 
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets. 
If you are generating double quotes as content of <question> or <answer>, make sure to escape them with a backslash. 
    """
    context.replace("{{TOPIC}}", topic)
    # agent_lm, agent_tokenizer, agent_client = agent_info

    context += f"Topic: {topic}\nAdditional requirements: {additional_req}\n"
    # extract the json file from the message
    
    # request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
    #                                  echo_prompt=False, temperature=0.0, max_tokens=4096,
    #                                  process_func=None, service=agent_client,
    #                                  terminate_by_linebreak='no', verbose=False)
    # response = request_result.completions[0].text

    response = gen_from_prompt(model=agent_info, prompt=context, temperature=0.0, max_tokens=4096)
    
    extracted_json = extract_json_v2(response, None)
    extracted_json = extracted_json[0]
    return extracted_json

def gen_qa_pairs_augmented(paragraph, agent_info, additional_req):
    context = """Conditioned on the wikipedia paragraph, you will generate a few question and answer pairs. 
Make sure not to ask subjective questions, and let the question's correct answer be a concise short phrase. 
Make sure that the question you selected is answerable by the given wikipedia paragraph, and make the answer concise. It's recommended to use the exact text from the paragraph as answers.
Make sure that the questions are also answerable by an expert **without the wikipedia paragraph**. For example, dont ask questions that are too specific to the paragraph, like "what are the three locations mentioned in the paragraph?". Or "who's the most famous soldier, according to the paragraph?".

You will also receive additional requirements on the questions. You should follow these additional requirements when generating the questions.
For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.

Try to generate a diverse set of 15 questions, and make sure that the questions are not too similar to each other while satisfying the additional requirements. If you can't generate 15 questions, generate as many as you can.

Formatting: 
Each question should be a dictionary with the following keys: id, question, answer, estimated difficulty. 
The questions should be exactly in the following format (a list of dictionaries): 
```json
{"id": "1", "question": "<question>", "answer": "<answer>", "difficulty": "1"}, 
{"id": "2", "question": "<question>", "answer": "<answer>", "difficulty": "1"}, 
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets. 
If you are generating double quotes as content of <question> or <answer>, make sure to escape them with a backslash. 
"""
    # agent_lm, agent_tokenizer, agent_client = agent_info

    context += f"Wiki paragraph: {paragraph}\nAdditional requirements: {additional_req}\n"
    # extract the json file from the message
    # request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
    #                                  echo_prompt=False, temperature=0.0, max_tokens=2000,
    #                                  process_func=None, service=agent_client,
    #                                  terminate_by_linebreak='no', verbose=False)
    # response = request_result.completions[0].text
    
    response = gen_from_prompt(model=agent_info, prompt=context, temperature=0.0, max_tokens=2000)

    extracted_json = extract_json_v2(response, None)
    extracted_json = extracted_json[0]
    return extracted_json






def _ask_question(theme, agent_info, history, iters, outfile_prefix='att1'):
    if os.path.exists(f"{outfile_prefix}.questions.json"):
        print("FOUND questions.json")
        return json.load(open(f"{outfile_prefix}.questions.json", "r"))
    # agent_lm, agent_tokenizer, agent_client = agent_info
    context = """
Your goal is to comprehensively evaluate the knowledge of a language model. 
In each iteration, you should output 30 ** knowledge-intensive ** questions of different categories, and write these questions in a json file. All the question need to pertain to the them of THEME.

To generate a question, you will follow the following steps: 
1. Come up with a category (e.g. physical phenomenon, capitals of countries, etc.)
2. This category will be used to search for a wikipedia page.
3. Conditioned on the wikipedia paragraph, you will generate a question and answer pair. Make sure that the question you selected is answerable by the given wikipedia paragraph.

Output formatting: 
Each question should be a dictionary with the following keys: id, question, answer, category, estimated difficulty.
Note: do not come up with repetitive questions. If you have asked a question, do not ask it again! 
Come up with 30 concrete questions, and write them in the following format. It's helpful to first come up with a plan for this iteration, and then write the questions.
The questions should be exactly in the following format (a list of dictionaries): 
```json
[
{"id": "1", "question": "What's the tallest mountain in the world?", "answer": "Mount Everest", "category": "Geography", "difficulty": "1"}, 
{"id": "2", "question": "Who discovered green tea", "answer":"Shen Nong", "category": "History", "difficulty": "1"},
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.

Iteration: 
The goal is to search for a category of questions that the language model is weak at. 
For iteration 1, you can start with questions of different categories, and start with a difficulty level of 1-2. Make sure the questions that you come up with are concrete questions that has a concrete solution, not just place holders, and come up with 30 questions. Do not leave place holders.  
In later iterations you should 
1. Think about breadth. Brainstorm questions with different categories if there are missing categories to make the evaluation more comprehensive and have broad coverage. 
2. For the categories that the model is strong at, increase the difficulty level of the questions. 
3. For the categories that the model is weak at, try to probe for diverse types of failure modes. Remember the goal is to get a comprehensive evaluation of the model. We want to know all the failure modes of the model, and all its strength.  
"""
    context = context.replace("THEME", theme)
    if iters is None:
        iters = len(history) + 1
    if iters == 1:
        context += "Please start with iteration 1."
    else:
        context += "\n".join(history) + "Please start with iteration {}. Remember: Always output 30 questions, DO NOT just terminate directly".format(iters)
    context = DEFAULT_JSON_MESSAGE + context
    # extract the json file from the message
    # request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
    #                                  echo_prompt=False, temperature=0.0, max_tokens=2000,
    #                                  process_func=None, service=agent_client,
    #                                  terminate_by_linebreak='no', )
    # response = request_result.completions[0].text
    
    response = gen_from_prompt(model=agent_info, prompt=context, temperature=0.0, max_tokens=2000)

    extracted_json = extract_json_v2(response, f"{outfile_prefix}.questions.json")
    return extracted_json


def generate_dataset_without_docs(line_, agent_info, outfile_prefix,
                            total_num_questions=50):
    if os.path.exists(f"{outfile_prefix}.KI_questions.json"):
        print("found ", f"{outfile_prefix}.KI_questions.json")
        full_lst = []
        with open(f"{outfile_prefix}.KI_questions.json", "r") as f:
            for line in f:
                line = json.loads(line)
                full_lst.append(line)
        return full_lst

    f = open(f"{outfile_prefix}.KI_questions.json", "w")

    full_lst = []

    try:
        json_questions = gen_qa_without_docs(line_['category'], agent_info, line_['additional_requirement'])
    except Exception as e:
        print(e)
        print("error in generating more questions, skipping...")
        print(f'generated {len(ful_lst)} questions')

    for json_question in json_questions:
        line = copy.deepcopy(line_)
        line['question'] = json_question['question']
        line['gold_answer'] = json_question['answer']
        line['difficulty'] = json_question['difficulty']
        line['wiki_entity'] = 'None'
        full_lst.append(line)
        print(json.dumps(line), file=f)
    f.close()
    return full_lst

def generate_long_questions(line_, agent_info, outfile_prefix, generate_qa_func=gen_qa_pairs_augmented,
                            total_num_questions=50):
    if os.path.exists(f"{outfile_prefix}.KI_questions.json"):
        print("found ", f"{outfile_prefix}.KI_questions.json")
        full_lst = []
        with open(f"{outfile_prefix}.KI_questions.json", "r") as f:
            for line in f:
                line = json.loads(line)
                full_lst.append(line)
        return full_lst

    f = open(f"{outfile_prefix}.KI_questions.json", "w")
    paragraph, wiki_entity, wiki_url = search_step(line_['category'], output_more=True)
    print(len(paragraph), 'length of paragraph')
    if len(paragraph) == 0:
        print("empty paragraph, skipping...")
        return {}

    full_lst = []
    for start_idx in range(0, len(paragraph), 20):
        if start_idx > total_num_questions: break
        end_idx = start_idx + 20 if start_idx + 20 < len(paragraph) else len(paragraph)
        try:
            json_questions = generate_qa_func(paragraph[start_idx:end_idx], agent_info, line_['additional_requirement'])
            # json_questions = generate_qa_func(paragraph[start_idx:end_idx], agent_info, line_['additional_requirement'])
        except:

            print("error in generating more questions, skipping...")
            print(f'generated {len(ful_lst)} questions')
            continue  # skip the empty paragraph.

        for json_question in json_questions:
            line = copy.deepcopy(line_)
            line['question'] = json_question['question']
            line['gold_answer'] = json_question['answer']
            line['difficulty'] = json_question['difficulty']
            line['wiki_entity'] = wiki_entity
            line['wiki_url'] = wiki_url
            full_lst.append(line)
            print(json.dumps(line), file=f)
    f.close()
    return full_lst


def saliency_rerank(json_lst, num_keep = 5 ):
    for line_ in json_lst:
        page_title = line_['category'].replace(' ', '_')
        pageviews = get_pageviews(page_title)
        line_['salience'] = pageviews  if pageviews is not None else 0 # add the pageviews to the line.
    # sort by the saliency
    json_lst = sorted(json_lst, key=lambda x: x['salience'], reverse=True)
    for line in json_lst:
        print(f'salience of category{line["category"]}: ', round(line['salience'] / 1000000, 2),  'M')
    return json_lst[:num_keep]



def generate_full_qa(theme, agent_info, history, iters, outfile_prefix='att1',
                                   historical_psg=None,
                                   category_gen_func=_refine_categories_targetacc_augmented,
                                   generate_qa_func=generate_long_questions, acc_target=None,
                                   apply_saliency_rerank=True):
    if os.path.exists(f"{outfile_prefix}.KI_questions.json"):
        print("FOUND KI_questions.json")
        return

    if acc_target is not None:
        json_category = category_gen_func(theme, agent_info, history, iters, outfile_prefix=outfile_prefix,
                                          acc_target=acc_target)
    else:
        json_category = category_gen_func(theme, agent_info, history, iters, outfile_prefix=outfile_prefix)
    if apply_saliency_rerank:
        json_category = saliency_rerank(json_category, 5)
    
    full_lst = []
    historical_psg = []
    for line_ in json_category:
        paragraph, wiki_entity, wiki_url = search_step(line_['category'])
        if wiki_entity in historical_psg:
            print('found repetitive wiki entity, skipping...', wiki_entity)
            continue
        if len(paragraph) == 0:
            print("empty paragraph, skipping...")
            continue  # skip the empty paragraph.
        historical_psg.append(wiki_entity)
        if 'additional_requirement' not in line_: continue # skip the empty paragraph.
        page_title = line_['category'].replace(' ', '_')
        pageviews = get_pageviews(page_title)
        line_['salience'] = pageviews if pageviews is not None else 0 # add the pageviews to the line.
        print(f'salience of page {page_title}: ', round(line_['salience'] / 1000000, 2), 'M')
        try:
            json_questions = generate_qa_func(line_, agent_info, outfile_prefix+f'__{page_title}')
        except Exception as e:
            print(e)
            print("error in generating questions, skipping...")
            continue # skip the empty paragraph.

        for line in json_questions:
            full_lst.append(line)
        line_['paragraph'] = paragraph
        line_['wiki_entity'] = wiki_entity
        line_['wiki_url'] = wiki_url

    with open(f"{outfile_prefix}.KI_questions.json", "w") as f:
        json.dump(full_lst, f)

    with open(f"{outfile_prefix}.categories_augmented.json", "w") as f:
        json.dump(json_category, f)
    return historical_psg


args = SimpleNamespace(num_iters=8, outfile_prefix1="data/video_games", theme="Video Games", acc_target="0.1--0.3")
history_dict = []
historical_psg = []
for iters in range(args.num_iters):
    args.outfile_prefix = args.outfile_prefix1 + str(iters + 1)
    summarized_content = summarize_over_history(history_dict, gold_key='gold_answer', verbose=False)
    history = [summarized_content]
    print(f"Iteration {iters}, SUMMARRY: {history}")
    historical_psg = generate_full_qa(args.theme, attacker_llm, history, iters + 1,
                               
                                      outfile_prefix=args.outfile_prefix,
                                                    historical_psg=historical_psg,
                                                    category_gen_func=_refine_categories_targetacc_augmented,
                                                    generate_qa_func=generate_long_questions,
                                                    acc_target=args.acc_target)
    with open(f"{args.outfile_prefix}.KI_questions.json", "r") as f:
        json_category = json.load(f)
    if len(json_category) == 1: # remove the outer embedded list.
        json_category = json_category[0]
    gold_answer_json = copy.deepcopy(json_category)
    test_taker_info = test_llm
    evaluator_info = test1_llm
    json_dict = solve_and_compare_questions(test_taker_info, evaluator_info, json_category, gold_answer_json,
                                            args.outfile_prefix, 'gold_answer')
    history_dict.append(json_dict)

    verbose_description = get_summary_of_results(json_dict, verbose=False)
    print(verbose_description)
    
    # After the main loop, implement the selection algorithm from the paper
print("\n" + "="*80)
print("Selecting Best Dataset Based on Optimization Objective")
print("="*80)

# Step 1: Load all generated datasets and compute metrics
all_datasets = {}
for iters in range(args.num_iters):
    iteration_file = args.outfile_prefix1 + str(iters + 1) + ".KI_questions.json"
    compare_file = args.outfile_prefix1 + str(iters + 1) + ".compare_answers.json"
    
    if os.path.exists(iteration_file) and os.path.exists(compare_file):
        with open(iteration_file, "r") as f:
            questions = json.load(f)
            if isinstance(questions, list) and len(questions) > 0 and isinstance(questions[0], list):
                questions = questions[0]
        
        with open(compare_file, "r") as f:
            evaluated = json.load(f)
        
        # Group by category (dataset description)
        for q in evaluated:
            category_key = q.get('category', 'Unknown')
            if 'wiki_entity' in q:
                category_key = f"{category_key}||{q['wiki_entity']}"
            
            if category_key not in all_datasets:
                all_datasets[category_key] = []
            all_datasets[category_key].append(q)

print(f"Found {len(all_datasets)} unique dataset descriptions")

# Step 2: Compute objective metrics for each dataset description
# Following Section 3.1: J(c) = NOVELTY + β1*DIFFICULTY + β2*SEP
beta1 = 1.0  # weight for difficulty
beta2 = 10.0  # weight for separability (as mentioned in paper Section 5.2)

dataset_scores = []
for category, questions in all_datasets.items():
    if len(questions) < 5:  # Skip categories with too few questions
        continue
    
    # Calculate DIFFICULTY: 1 - max_accuracy
    accuracies = [1 if q['is_correct'] == 'true' else 0 for q in questions]
    difficulty = 1 - (sum(accuracies) / len(accuracies))
    
    # Calculate SEPARABILITY: mean absolute deviation
    # (simplified version - in paper they compute across multiple models)
    mean_acc = sum(accuracies) / len(accuracies)
    separability = sum([abs(a - mean_acc) for a in accuracies]) / len(accuracies)
    
    # For NOVELTY, we would need to compare against prior benchmarks
    # For simplicity, we'll use difficulty as a proxy (harder = more novel)
    novelty = difficulty
    
    # Compute objective score (Equation 1 from paper)
    objective_score = novelty + beta1 * difficulty + beta2 * separability
    
    dataset_scores.append({
        'category': category,
        'num_questions': len(questions),
        'difficulty': difficulty,
        'separability': separability,
        'novelty': novelty,
        'objective_score': objective_score,
        'questions': questions
    })

# Step 3: Sort by objective score and select top datasets
dataset_scores.sort(key=lambda x: x['objective_score'], reverse=True)

print("\nDataset Descriptions Ranked by Objective Score:")
print("-" * 80)
for i, ds in enumerate(dataset_scores[:10]):
    print(f"{i+1}. {ds['category']}")
    print(f"   Questions: {ds['num_questions']}, Difficulty: {ds['difficulty']:.3f}, "
          f"Sep: {ds['separability']:.3f}, Objective: {ds['objective_score']:.3f}")

# Step 4: Select questions from top-ranked dataset descriptions
# Select 10-20 questions per theme (you can adjust this)
QUESTIONS_PER_CATEGORY = 15
NUM_TOP_CATEGORIES = min(5, len(dataset_scores))  # Select top 5 categories

selected_questions = []
for ds in dataset_scores[:NUM_TOP_CATEGORIES]:
    # Randomly sample questions from this category
    category_questions = ds['questions']
    sample_size = min(QUESTIONS_PER_CATEGORY, len(category_questions))
    sampled = random.sample(category_questions, sample_size)
    selected_questions.extend(sampled)
    print(f"\nSelected {sample_size} questions from: {ds['category']}")

# Step 5: Save the final selected dataset
final_dataset_path = args.outfile_prefix1 + "_final_selected_dataset.json"
with open(final_dataset_path, "w") as f:
    json.dump(selected_questions, f, indent=2)

print(f"\n{'='*80}")
print(f"✓ SUCCESS: Final selected dataset saved to {final_dataset_path}")
print(f"✓ Total selected questions: {len(selected_questions)}")
print(f"✓ From {NUM_TOP_CATEGORIES} top-ranked categories")
print(f"{'='*80}\n")

# Step 6: Save metadata about selection
metadata = {
    'selection_criteria': 'Maximized: NOVELTY + β1*DIFFICULTY + β2*SEPARABILITY',
    'beta1': beta1,
    'beta2': beta2,
    'num_categories_selected': NUM_TOP_CATEGORIES,
    'questions_per_category': QUESTIONS_PER_CATEGORY,
    'total_questions': len(selected_questions),
    'top_categories': [
        {
            'rank': i+1,
            'category': ds['category'],
            'objective_score': ds['objective_score'],
            'difficulty': ds['difficulty'],
            'num_questions_selected': min(QUESTIONS_PER_CATEGORY, len(ds['questions']))
        }
        for i, ds in enumerate(dataset_scores[:NUM_TOP_CATEGORIES])
    ]
}

metadata_path = args.outfile_prefix1 + "_selection_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Selection metadata saved to: {metadata_path}")