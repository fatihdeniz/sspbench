import os
from src.utils import colored_print as print
from .exaggerate_generator import HarmfulWordProcessor,QuestionGenerator
from .jailbreak_generator import HarmfulQueryGenerator, Test_Case_Builder   


def run(base_dir=None):
    print("Running Jailbreak Generator ...")
    generator = HarmfulQueryGenerator(base_path=base_dir)
    generator.harmful_query_generation('metadata/harmful_queries_1.json', 'metadata/generated_raw_data/safety', max_workers=10)

    generator = Test_Case_Builder(base_path=base_dir,
                                  save_path='llm_safety/all_jailbreak_prompts.json')

    generator.jailbreak_add_attack(jailbreak_method='ALL', number_for_each_topic=1, save_to_file=True)
    
    print("Running Exaggerate Generator ...")
    processor = HarmfulWordProcessor(base_path=base_dir)
    processor.run_pipeline()
    
    question_generator = QuestionGenerator(base_path=base_dir)
    question_generator.run()
    
