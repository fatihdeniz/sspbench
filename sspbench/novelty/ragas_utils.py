from typing import List, Any
import numpy as np


# =========================
# Sentence Transformers Embedding Model
# =========================

class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


# =========================
# RAGAS availability check
# =========================

def is_ragas_available() -> bool:
    try:
        import ragas
        return True
    except ImportError:
        return False


# =========================
# RAGAS LLM adapter
# =========================

from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig


class Generation:
    def __init__(self, text: str):
        self.text = text
    
    def __str__(self):
        return self.text


class LLMResult:
    def __init__(self, text: str, n: int = 1):
        gen = Generation(text)
        self.generations = [[gen] for _ in range(n)]


class CustomRagasLLM(BaseRagasLLM):
    def __init__(
        self,
        model: Any,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.run_config = RunConfig()

    async def generate(self, messages, n: int = 1, **kwargs):
        """Async method to handle RAGAS calls with await."""
        from .llm_utils import gen_from_prompt
        
        system_prompt = "You are a helpful assistant."
        user_content = None
        
        if hasattr(messages, 'text'):
            user_content = messages.text
        elif isinstance(messages, tuple) and len(messages) >= 2:
            if messages[0] == 'text':
                user_content = messages[1]
        elif isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    
                    if isinstance(content, tuple) and len(content) >= 2:
                        content = content[1]
                    elif hasattr(content, 'text'):
                        content = content.text
                    
                    if role == 'system':
                        system_prompt = content if content else system_prompt
                    elif role == 'user':
                        user_content = content
                elif isinstance(msg, tuple) and len(msg) >= 2:
                    if msg[0] == 'text':
                        user_content = msg[1]
        
        if not user_content:
            user_content = ""
        
        text = gen_from_prompt(
            self.model,
            user_content,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=system_prompt
        )
        if not isinstance(text, str):
            text = str(text)
        
        return LLMResult(text, n)

    def generate_text(self, prompt: str, n: int = 1, **kwargs) -> LLMResult:
        from .llm_utils import gen_from_prompt
        
        if isinstance(prompt, tuple) and len(prompt) >= 2:
            prompt = prompt[1]
            
        text = gen_from_prompt(
            self.model,
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if not isinstance(text, str):
            text = str(text)
        return LLMResult(text, n)

    async def agenerate_text(self, prompt: str, n: int = 1, **kwargs) -> LLMResult:
        return self.generate_text(prompt, n, **kwargs)

    def is_finished(self, text: str) -> bool:
        return True


# =========================
# RAGAS Embeddings adapter
# =========================

from ragas.embeddings import BaseRagasEmbeddings


class CustomRagasEmbeddings(BaseRagasEmbeddings):
    def __init__(self, embedding_model: Any):
        self.embedding_model = embedding_model
        self.run_config = RunConfig()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


# =========================
# Q&A generation (RAGAS 0.4.3)
# =========================

def generate_qa_with_ragas(
    paragraph: str,
    agent_info: Any,
    embedding_model: Any = None,
    num_questions: int = 3,
) -> List[dict]:
    from ragas.testset import TestsetGenerator

    if embedding_model is None:
        if not hasattr(agent_info, "embedding_model"):
            raise ValueError(
                "RAGAS 0.4.3 requires an embedding model. "
                "Pass `embedding_model=...` explicitly or attach "
                "`embedding_model` to agent_info."
            )
        embedding_model = agent_info.embedding_model

    ragas_llm = CustomRagasLLM(agent_info, temperature=0.7)
    ragas_embeddings = CustomRagasEmbeddings(embedding_model)

    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
    )

    from langchain_core.documents import Document
    doc = Document(page_content=paragraph)
    
    testset = generator.generate_with_langchain_docs(
        documents=[doc],
        testset_size=num_questions,
    )

    qa_pairs = []
    for idx, sample in enumerate(testset.samples, 1):
        qa_pairs.append(
            {
                "id": str(idx),
                "question": sample.eval_sample.user_input,
                "answer": sample.eval_sample.reference,
                "difficulty": "2",
            }
        )

    return qa_pairs



# =========================
# Faithfulness evaluation
# =========================

def evaluate_qa_faithfulness(
    question: str,
    answer: str,
    context: str,
    eval_model: Any,
) -> dict:
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas import evaluate
    from datasets import Dataset

    ragas_llm = CustomRagasLLM(eval_model, temperature=0.0)

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [[context]],
        "ground_truth": [answer],
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
    )

    return {
        "faithfulness": float(result["faithfulness"][0]),
        "answer_relevancy": float(result["answer_relevancy"][0]),
    }
