from abc import ABC, abstractmethod

class OrchestratorBase(ABC):
    def __init__(self, args, llm):
        self.args = args
        self.llm = llm

    @abstractmethod
    def run(self):
        pass