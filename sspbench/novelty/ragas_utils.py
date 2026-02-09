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
    def __init__(self, texts):
        if isinstance(texts, list):
            self.generations = [[Generation(text)] for text in texts]
        else:
            self.generations = [[Generation(texts)]]


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
        """Sync method to handle RAGAS calls."""
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
        
        if n == 1:
            text = gen_from_prompt(
                self.model,
                user_content,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                system_prompt=system_prompt
            )
            if not isinstance(text, str):
                text = str(text)
            return LLMResult([text])
        else:
            # Generate multiple completions with prompt variations and temperature for diversity
            texts = []
            base_temp = self.temperature
            
            # Different prompt variations to encourage diversity
            prompt_variations = [
                user_content,  # original
                f"Please answer this question: {user_content}",  # more formal
                f"Respond to: {user_content}"  # different framing
            ]
            
            for i in range(n):
                # Use prompt variation and temperature variation
                varied_prompt = prompt_variations[i % len(prompt_variations)]
                temp_variation = [0.1, 0.7, 1.5][i % 3] if n >= 3 else base_temp
                
                text = gen_from_prompt(
                    self.model,
                    varied_prompt,
                    temperature=temp_variation,
                    max_tokens=self.max_tokens,
                    system_prompt=system_prompt
                )
                if not isinstance(text, str):
                    text = str(text)
                texts.append(text)
            return LLMResult(texts)

    def generate_text(self, prompt: str, n: int = 1, **kwargs) -> LLMResult:
        from .llm_utils import gen_from_prompt
        
        if isinstance(prompt, tuple) and len(prompt) >= 2:
            prompt = prompt[1]
        
        if n == 1:
            text = gen_from_prompt(
                self.model,
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if not isinstance(text, str):
                text = str(text)
            return LLMResult([text])
        else:
            texts = []
            base_temp = self.temperature
            
            prompt_variations = [
                prompt,  # original
                f"Please answer this question: {prompt}",  # more formal
                f"Respond to: {prompt}"  # different framing
            ]
            
            for i in range(n):
                varied_prompt = prompt_variations[i % len(prompt_variations)]
                temp_variation = [0.1, 0.7, 1.5][i % 3] if n >= 3 else base_temp
                
                text = gen_from_prompt(
                    self.model,
                    varied_prompt,
                    temperature=temp_variation,
                    max_tokens=self.max_tokens,
                )
                if not isinstance(text, str):
                    text = str(text)
                texts.append(text)
            return LLMResult(texts)

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
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
    from ragas.testset.transforms import HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor, apply_transforms
    from ragas.testset.transforms.extractors import NERExtractor
    from ragas.testset.persona import Persona

    if embedding_model is None:
        if not hasattr(agent_info, "embedding_model"):
            raise ValueError("Pass `embedding_model=...` explicitly.")
        embedding_model = agent_info.embedding_model

    ragas_llm = CustomRagasLLM(agent_info, temperature=0.7)
    ragas_embeddings = CustomRagasEmbeddings(embedding_model)

    # Configure query distribution for ONLY short, specific questions
    query_distribution = [(SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 1.0)]

    
    kg = KnowledgeGraph()
    from langchain_core.documents import Document
    doc = Document(page_content=paragraph, metadata={"source": "novelty_engine"})

    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )
    
    personas = [
        Persona(
            name="Knowledge_Seeker",
            role_description=(
                "Asks single-hop factual trivia questions with short answers. "
                "Focuses on concrete facts such as names, dates, places, or quantities."
                "Do not ask “why”, “explain”, “role”, “how”. "
                "Produce only short factoid questions and answers."
            ),
        ),
        Persona(
            name="Expert_Inquirer",
            role_description=(
                "Asks difficult but single-hop factual questions similar to TriviaQA "
                "or quiz-bowl factoids. Requires precise, unambiguous answers."
                "Answer must be a single named entity (person name, year, organization, location, quantity, etc.) with no explanation. "
                "Do not ask “why”, “explain”, “role”, “how”. "
                "Produce only short factoid questions and answers."
            ),
        ),
        Persona(
            name="Casual_User",
            role_description=(
                "Asks straightforward factual questions in natural language. "
                "Answer must be a single named entity (person name, year, organization, location, quantity, etc.) with no explanation. "
                "Do not ask “why”, “explain”, “role”, “how”. "
                "Produce only short factoid questions and answers."
            ),
        ),
    ]
    
    # keyphrase_extractor = KeyphrasesExtractor(llm=ragas_llm)
    ner_extractor = NERExtractor(llm=ragas_llm)
    transforms = [ner_extractor]
    apply_transforms(kg, transforms=transforms)

    generator = TestsetGenerator(
            llm=ragas_llm,
            embedding_model=ragas_embeddings,
            knowledge_graph=kg,
            persona_list=personas
        )

    
    testset = generator.generate_with_langchain_docs(
        documents=[doc],
        testset_size=num_questions,
        query_distribution=query_distribution,
    )

    qa_pairs = []
    for idx, sample in enumerate(testset.samples, 1):
        print(f"Generated Sample {idx}:", sample.eval_sample.user_input)
        qa_pairs.append(
            {
                "id": str(idx),
                "question": sample.eval_sample.user_input,
                "answer": sample.eval_sample.reference,
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
    eval_model: BaseRagasLLM,
    embedding_model: BaseRagasEmbeddings = None,
) -> dict:
    """
    Evaluate faithfulness and answer relevancy using RAGAS.

    Note: This function uses the deprecated evaluate() API.
    Consider using the @experiment decorator for future implementations.

    Args:
        question: The question to evaluate
        answer: The answer to evaluate
        context: The context paragraph
        eval_model: The LLM to use for evaluation
        embedding_model: The embeddings model (optional)

    Returns:
        Dictionary with faithfulness and answer_relevancy scores
    """
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas import evaluate
    from datasets import Dataset
    from ragas.run_config import RunConfig

    ragas_llm = CustomRagasLLM(eval_model, temperature=0.0)
    ragas_embeddings = CustomRagasEmbeddings(embedding_model) if embedding_model else None

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [[context]],
        "ground_truth": [answer],
    }

    dataset = Dataset.from_dict(data)

    # Use the deprecated evaluate() function with proper parameters
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(),
        raise_exceptions=False,
        show_progress=False,
    )

    # Handle EvaluationResult object - access scores as lists
    return {
        "faithfulness": float(result["faithfulness"][0]) if result["faithfulness"] else 0.0,
        "answer_relevancy": float(result["answer_relevancy"][0]) if result["answer_relevancy"] else 0.0,
    }
