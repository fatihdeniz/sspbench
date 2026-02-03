"""
Jailbroken Class
============================================
Jailbroken utilized competing objectives and mismatched generalization 
modes of LLMs to constructed 29 artificial jailbreak methods.

Paper title: Jailbroken: How Does LLM Safety Training Fail?
arXiv Link: https://arxiv.org/pdf/2307.02483.pdf
"""
import json
import logging
from collections import defaultdict
logging.basicConfig(level=logging.WARNING)

from easyjailbreak.metrics.Evaluator import EvaluatorGenerativeJudge
from easyjailbreak.attacker import AttackerBase
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.mutation.rule import *
from conversation import Conversation

__all__ = ['Jailbroken']

MUTATION_CATEGORY_MAP = defaultdict(
    lambda: "N/A",
    {
        "Artificial": "Adversarial prefix prompts",
        "Base64": "Entire request Base64-encoded",
        "Base64_input_only": "User input Base64-encoded",
        "Base64_raw": "Inline Base64 fragments",
        "Combination_1": "Base64 + malicious prefix",
        "Combination_2": "Combination 1 + stylistic constraints",
        "Combination_3": "Combination 2 + Wikipedia mimicry",
        "Disemvowel": "Vowel-removed obfuscation",
        "Leetspeak": "Leetspeak character substitutions",
        "Rot13": "ROT13 letter substitution",
        "Auto_payload_splitting": "Automated payload splitting",
        "Auto_obfuscation": "Automated semantic obfuscation",
    }
)

class Jailbroken(AttackerBase):
    r"""
    Implementation of Jailbroken Jailbreak Challenges in Large Language Models
    """
    def __init__(self, attack_model, target_model, eval_model, jailbreak_datasets: JailbreakDataset, batch_size=4):
        r"""
        :param attack_model: The attack_model is used to generate the adversarial prompt.
        :param target_model: The target language model to be attacked.
        :param eval_model: The evaluation model to evaluate the attack results.
        :param jailbreak_datasets: The dataset to be attacked.
        :param template_file: The file path of the template.
        """
        super().__init__(attack_model, target_model, eval_model, jailbreak_datasets)
        self.mutations = [
            Artificial(attr_name='query'),
            Base64(attr_name='query'),
            Base64_input_only(attr_name='query'),
            Base64_raw(attr_name='query'),
            Disemvowel(attr_name='query'),
            Leetspeak(attr_name='query'),
            Rot13(attr_name='query'),
            Combination_1(attr_name='query'),
            Combination_2(attr_name='query'),
            Combination_3(attr_name='query'),
            Auto_payload_splitting(self.attack_model, attr_name='query', eval_model=self.eval_model),
            Auto_obfuscation(self.attack_model, attr_name='query', eval_model=self.eval_model),

        ]
        self.evaluator = EvaluatorGenerativeJudge(eval_model)
        self.current_jailbreak = 0
        self.current_query = 0
        self.current_reject = 0
        self.batch_size = batch_size

    def single_attack(self, instance: Instance) -> JailbreakDataset:
        r"""
        single attack process using provided prompts and mutation methods.

        :param instance: The Instance that is attacked.
        """
        instance_ds = JailbreakDataset([instance])
        source_instance_list = []
        updated_instance_list = []

        print(f"Base: {instance}")
        for mutation in self.mutations:
            transformed_jailbreak_datasets = mutation(instance_ds)
            print(f"Mutation: {mutation}, Sample count: {len(transformed_jailbreak_datasets)}")
            for item in transformed_jailbreak_datasets:
                source_instance_list.append(item)
                    
        for instance in source_instance_list:
            answer = self.target_model.generate(instance.jailbreak_prompt.format(query = instance.query))[0]
            instance.target_responses.append(answer)
            updated_instance_list.append(instance)
        return JailbreakDataset(updated_instance_list)
    
    def batch_attack(self, instances) -> JailbreakDataset:
        if isinstance(instances, Instance):
            instances = [instances]

        source_instance_list = []
        updated_instance_list = []

        for instance in instances:
            base_ds = JailbreakDataset([instance])
            for mutation in self.mutations:
                transformed_jailbreak_datasets = mutation(base_ds)
                for item in transformed_jailbreak_datasets:
                    item["attack_attrs"]["Mutation"] = mutation.__class__.__name__
                    source_instance_list.append(item)

        if not source_instance_list:
            return JailbreakDataset([])

        conversations = []
        for inst in source_instance_list:
            conv = Conversation()
            conv.add_message("user", inst.jailbreak_prompt.format(query=inst.query))
            conversations.append(conv)
        
        _, answers = self.target_model.generate_conversation(conversations)

        for inst, ans in zip(source_instance_list, answers):
            inst.target_responses.append(ans)
            updated_instance_list.append(inst)

        return JailbreakDataset(updated_instance_list)

    def generate_results(self, out_file):
        instance_count = len(self.attack_results)
        print("Processed instance #:", instance_count)

        results = []

        for index in range(instance_count):
            instance = self.attack_results[index]
            prompt = instance.query
            conversation = instance.to_single_conv().to_list()

            result = {
                "prompt": prompt,
                "conversation": conversation,
                "jailbreak_prompt": conversation[-2]["content"],
                "response": conversation[-1]["content"],
                "eval_results": instance.eval_results[-1],
                "category": instance["attack_attrs"]["Mutation"],
                "categoryDescription": MUTATION_CATEGORY_MAP[instance["attack_attrs"]["Mutation"]]
            }
            results.append(result)

        with open(out_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"Saved extracted results to {out_file}")
        return results
    
    def update(self, Dataset: JailbreakDataset):
        r"""
        Update the state of the Jailbroken based on the evaluation results of Datasets.
        
        :param Dataset: The Dataset that is attacked.
        """
        for prompt_node in Dataset:
            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject
            
    def attack(self):
        r"""
        Execute the attack process using provided prompts and mutations.
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
        self.update(self.attack_results)
        logging.info("Jailbreak finished!")

    def log(self):
        r"""
        Report the attack results.
        """
        logging.info("======Jailbreak report:======")
        logging.info(f"Total queries: {self.current_query}")
        logging.info(f"Total jailbreak: {self.current_jailbreak}")
        logging.info(f"Total reject: {self.current_reject}")
        logging.info("========Report End===========")
