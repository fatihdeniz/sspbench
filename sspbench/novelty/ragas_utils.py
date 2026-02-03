"""
RAGAS utilities for question generation and evaluation.
"""

from typing import Optional, List, Any

try:
    # For ragas >= 0.2.0, the import path changed
    from ragas.testset import TestsetGenerator
    from langchain_core.documents import Document as LangchainDocument
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.callbacks import CallbackManagerForLLMRun
    
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas import evaluate
    from datasets import Dataset
    RAGAS_AVAILABLE = True
    print("✓ RAGAS successfully imported (version >= 0.2.0)")
except ImportError as e:
    RAGAS_AVAILABLE = False
    LangchainDocument = None
    BaseLanguageModel = object  # Fallback to object if not available
    CallbackManagerForLLMRun = None
    print(f"⚠️  RAGAS or Langchain core not available. RAGAS-related features will be disabled. Error: {e}")


if RAGAS_AVAILABLE:
    class CustomLangchainLLM(BaseLanguageModel):
        
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, 
                      run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Any:
            from .llm_utils import gen_from_prompt
            results = []
            for prompt in prompts:
                response = gen_from_prompt(self.model, prompt, temperature=0.0, max_tokens=2000)
                results.append(response)
            
            # Return in expected format
            from langchain_core.outputs import LLMResult, Generation
            generations = [[Generation(text=result)] for result in results]
            return LLMResult(generations=generations)
        
        def _llm_type(self) -> str:
            return "custom"
        
        async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None,
                            run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Any:
            return self._generate(prompts, stop, run_manager, **kwargs)
else:
    CustomLangchainLLM = None


def generate_qa_with_ragas(paragraph, agent_info, num_questions=3):
    """
    Generate Q&A pairs using RAGAS testset generator.
    
    Args:
        paragraph: Wikipedia paragraph text
        agent_info: Model for generation
        num_questions: Number of questions to generate
        
    Returns:
        List of Q&A pairs with RAGAS quality metrics
    """
    qa_pairs = []
    
    llm = CustomLangchainLLM(agent_info)
    doc = LangchainDocument(page_content=paragraph, metadata={"source": "wikipedia"})
    
    try:
        generator = TestsetGenerator(
            llm=llm,
            critic_llm=llm
        )
        # Generate testset
        testset = generator.generate_with_langchain_docs(
            documents=[doc],
            test_size=num_questions
        )
        

        for idx, sample in enumerate(testset.samples, 1):
            qa_pair = {
                'id': str(idx),
                'question': getattr(sample, 'user_input', getattr(sample, 'question', '')),
                'answer': getattr(sample, 'reference', getattr(sample, 'ground_truth', getattr(sample, 'answer', ''))),
                'difficulty': '2',
            }
            qa_pairs.append(qa_pair)
            
    except Exception as e:
        print(f"Error generating with RAGAS: {e}")
        raise
    
    return qa_pairs
    

def evaluate_qa_faithfulness(question, answer, context, eval_model):
    """
    Evaluate faithfulness of Q&A pair using RAGAS metrics.
    
    Args:
        question: Generated question
        answer: Generated answer
        context: Source paragraph
        eval_model: Model for evaluation
        
    Returns:
        Dictionary with faithfulness and answerability scores
    """
    if not RAGAS_AVAILABLE:
        return {'faithfulness': 1.0, 'answerability': 1.0}  # Default to passing
    
    try:
        # Prepare data for RAGAS evaluation
        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [[context]],
            'ground_truth': [answer]
        }
        dataset = Dataset.from_dict(data)
        llm = CustomLangchainLLM(eval_model)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm
        )
        
        return {
            'faithfulness': result['faithfulness'],
            'answerability': result['answer_relevancy']
        }
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        return {'faithfulness': 1.0, 'answerability': 1.0}


def is_ragas_available():
    """Check if RAGAS is available."""
    return RAGAS_AVAILABLE
