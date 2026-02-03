"""
Cipher Class
============================================
This Class enables humans to chat with LLMs through cipher prompts topped with 
system role descriptions and few-shot enciphered demonstrations.

Paper title：GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher
arXiv Link: https://arxiv.org/pdf/2308.06463.pdf
Source repository: https://github.com/RobustNLP/CipherChat
"""
import logging
logging.basicConfig(level=logging.WARNING)
import pandas as pd
import json
from easyjailbreak.metrics.Evaluator import EvaluatorGenerativeJudge
from easyjailbreak.attacker import AttackerBase
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.mutation.rule import MorseExpert, CaesarExpert, AsciiExpert, SelfDefineCipher
from conversation import Conversation
from utils import clean_response

MUTATION_CATEGORY_MAP = {
        "MorseExpert": "Morse",
        "CaesarExpert": "Caesar",
        "AsciiExpert": "Ascii",
        "SelfDefineCipher": "Self Defined",
    }

MUTATION_SUBCATEGORY_MAP = {
        "illegal": "Illegal",
        "ethics": "Ethics",
        "opinion": "Opinion",
        "insult": "Insult",
        "mental": "Mental",
        "physical": "Physical",
        "privacy": "Privacy",
        "exposure": "Exposure",
        "unfairness": "Fairness",
        "unsafe": "Safety",
    }

__all__ = ['Cipher']

class Cipher(AttackerBase):
    r"""
    Cipher is a class for conducting jailbreak attacks on language models. It integrates attack
    strategies and policies to evaluate and exploit weaknesses in target language models.
    """
    def __init__(self, attack_model, target_model, eval_model, jailbreak_datasets: JailbreakDataset, batch_size=3):
        super().__init__(attack_model, target_model, eval_model, jailbreak_datasets)
        r"""
        Initialize the Cipher Attacker.
        :param attack_model: In this case, the attack_model should be set as None.
        :param target_model: The target language model to be attacked.
        :param eval_model: The evaluation model to evaluate the attack results.
        :param jailbreak_datasets: The dataset to be attacked.
        """
        self.mutations = {
            "MorseExpert": MorseExpert(eval_model=eval_model),
            "CaesarExpert": CaesarExpert(eval_model=eval_model),
            "AsciiExpert": AsciiExpert(eval_model=eval_model),
            "SelfDefineCipher": SelfDefineCipher(eval_model=eval_model)
        }
        self.evaluator = EvaluatorGenerativeJudge(eval_model)
        self.info_dict = {'query': []}
        self.info_dict.update({expert.__class__.__name__: [] for expert in self.mutations.values()})
        self.df = None
        self.batch_size = batch_size

    
    def batch_attack(self, instances) -> JailbreakDataset:
        if isinstance(instances, Instance):
            instances = [instances]

        source_instance_list = []
        updated_instance_list = []

        for instance in instances:
            for mutation in self.mutations.values():
                for subcategory in MUTATION_SUBCATEGORY_MAP.keys():
                    instance.attack_attrs["Mutation"] = mutation.__class__.__name__
                    instance.attack_attrs['query_class'] = subcategory
                    base_ds = JailbreakDataset([instance])

                    transformed_jailbreak_datasets = mutation(base_ds)
                    for item in transformed_jailbreak_datasets:
                        source_instance_list.append(item)

        if not source_instance_list:
            return JailbreakDataset([])

        conversations = []
        for inst in source_instance_list:
            conv = Conversation()
            conv.add_message("user", inst.jailbreak_prompt.format(encoded_query = inst.encoded_query))
            conversations.append(conv)
        
        _, answers = self.target_model.generate_conversation(conversations)

        for inst, ans in zip(source_instance_list, answers):
            clean_ans = clean_response(ans)
            inst.raw_response = ans
            inst.encoded_target_responses = clean_ans
            inst.target_responses.append(self.mutations[inst.attack_attrs["Mutation"]].decode(clean_ans))
            updated_instance_list.append(inst)

        return JailbreakDataset(updated_instance_list)
    
    def attack(self):
        r"""
        Execute the attack process using four cipher methods on the entire jailbreak_datasets.
        """
        logging.info("Jailbreak started!")
        self.attack_results = JailbreakDataset([])
        try:
            
            for block in self.jailbreak_datasets.chunks(self.batch_size):
                results = self.batch_attack(block)
                for inst in results:
                    self.attack_results.add(inst)

        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")
        
        self.evaluator(self.attack_results)
        logging.info("Jailbreak finished!")
        

    def generate_results(self, out_file):
        instance_count = len(self.attack_results)
        print("Processed instance #:", instance_count)

        results = []

        for index in range(instance_count):
            instance = self.attack_results[index]
            prompt = instance.query
            
            conv = Conversation()
            conv.add_message("user", instance.jailbreak_prompt.format(encoded_query=instance.encoded_query))
            conv.add_message('assistant', instance.target_responses[-1])
            conv = conv.to_list()
            result = {
                "prompt": prompt,
                "conversation": conv,
                "jailbreak_prompt": conv[-2]["content"],
                "response": conv[-1]["content"],
                "raw_response": instance["raw_response"],
                "eval_results": instance.eval_results[-1],
                "category": MUTATION_CATEGORY_MAP[instance.attack_attrs["Mutation"]],
                "subcategory": MUTATION_SUBCATEGORY_MAP[instance.attack_attrs['query_class']],
            }
            results.append(result)

        with open(out_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"Saved extracted results to {out_file}")
        return results
        