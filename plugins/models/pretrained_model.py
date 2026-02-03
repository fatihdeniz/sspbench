import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.llm_base import LLMBase

class PretrainedLLM(LLMBase):
    def __init__(self, model_path):
        super().__init__(model_path)

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.to(self.device)
            self.model.half()
        else:
            self.device = torch.device('cpu')

    def generate(self, batch, sampling_params):
        batch = self.normalize_batch(batch)
            
        inputs = self.tokenizer(batch, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, 
                                max_length=sampling_params.max_tokens)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **sampling_params)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def check_role_support(self, role):
        conversation = [{"role": role, "content": "Test message"}]
        try:
            self.tokenizer(conversation[0]["content"], return_tensors="pt")
            return True
        except Exception as e:
            print(f"Failed to apply chat template with exception {e}")
            return False
