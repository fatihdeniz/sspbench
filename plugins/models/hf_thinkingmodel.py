from models.hf_model import HuggingFaceLLM

class HuggingFaceThinkingLLM(HuggingFaceLLM):
    def __init__(self, model_path, gpu_memory_utilization=0.85, max_model_len=4096, quantization=None):
        super().__init__(model_path, gpu_memory_utilization, max_model_len, quantization)
        self.apply_chat_template = False

    def generate(self, batch, sampling_params):
        sampling_params.max_tokens = 3072        
        
        for messages in batch:
            for message in messages:
                if message['role'] == 'user':
                    message['role'] = 'thinking_user'
        
        initial_responses = self.get_responses(batch, sampling_params)

        resubmission_indices = []
        resubmission_batch = []
        for i, output in enumerate(initial_responses):
            if '</think>' not in output:
                for message in batch[i]:
                    if message['role'] == 'thinking_user':
                        message['role'] = 'user'
                batch[i] += [{'role': 'thinking', 'content': output}]
                resubmission_indices.append(i)
                resubmission_batch.append(batch[i])
        if resubmission_batch:
            sampling_params.max_tokens = 1024
            resubmission_responses = self.get_responses(resubmission_batch, sampling_params)

            for idx, resubmitted_output in zip(resubmission_indices, resubmission_responses):
                initial_responses[idx] = f'{initial_responses[idx]}\n</think>\n{resubmitted_output}'

        return [f'<think>\n{response}' for response in initial_responses]

    def get_responses(self, batch, sampling_params):
        queries = [self.tokenizer.apply_chat_template(messages, tokenize=False) for messages in batch]
        batch_output = self.llm.generate(queries, sampling_params)
        return [output.outputs[0].text.strip() for output in batch_output]
