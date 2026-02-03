from judge_models import ATTACKER_MODEL, JUDGE_MODELS
from models.llm_factory import LLMFactory
from easyjailbreak.attacker import PAIR
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.orchestrator import OrchestratorBase

class PairOrchestrator(OrchestratorBase):
    def __init__(self, args, llm):
        self.args = args
        self.llm = llm

    def run(self):
        dataset = JailbreakDataset(self.args.input, local_file_type="json")
        attack_llm = LLMFactory.from_config(ATTACKER_MODEL)
        eval_llm = LLMFactory.from_config(JUDGE_MODELS[self.args.test])
        attacker = PAIR(attack_model=attack_llm,
                        target_model=self.llm,
                        eval_model=eval_llm,
                        jailbreak_datasets=dataset,
                    batch_size=20)
        return attacker.attack(save_path=self.args.output)
