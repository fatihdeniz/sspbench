from typing import List, Dict, Set, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter

from ..utils.llm_utils import gen_from_prompt
from .quality import evaluate_question_quality
from .config import QUALITY_THRESHOLD, MAX_REGENERATION_ROUNDS, FEEDBACK_SUMMARY_LIMIT

def is_ragas_available() -> bool:
    try:
        import ragas
        return True
    except ImportError:
        return False

from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig
from ragas.embeddings import BaseRagasEmbeddings
# =========================
# Sentence Transformers Embedding Model
# =========================
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()

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
        
        if isinstance(prompt, tuple) and len(prompt) >= 2:
            prompt = prompt[1]
        
        texts = []
        for i in range(n):
            # Use temperature variation to get diverse outputs
            temp = self.temperature + (i * 0.2) if n > 1 else self.temperature
            temp = min(temp, 2.0)  # Cap at 2.0
            
            text = gen_from_prompt(
                self.model,
                prompt,
                temperature=temp,
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
# Q&A generation
# =========================
def generate_qa_with_ragas(
    paragraph: str,
    agent_info: Any,
    embedding_model: Any = None,
    num_questions: int = 3,
    eval_model: Any = None,
    quality_threshold: int = None,
    max_regeneration_rounds: int = None,
) -> List[dict]:
    """
    Generate short, single-hop, entity-grounded factual QA pairs with quality control.

    Output questions are factoid-style and answers are atomic entities
    (name, date, location, organization, quantity, etc.).
    
    Args:
        paragraph: Source text for question generation
        agent_info: Model for generation
        embedding_model: Embedding model (optional)
        num_questions: Target number of questions to generate
        eval_model: Model for quality evaluation (optional, enables quality control)
        quality_threshold: Minimum quality score (default: QUALITY_THRESHOLD from config)
        max_regeneration_rounds: Maximum regeneration attempts (default: MAX_REGENERATION_ROUNDS from config)
        
    Returns:
        List of QA dictionaries with quality metadata if eval_model is provided
    """
    # Use config defaults if not specified
    if quality_threshold is None:
        quality_threshold = QUALITY_THRESHOLD
    if max_regeneration_rounds is None:
        max_regeneration_rounds = MAX_REGENERATION_ROUNDS

    from langchain_core.documents import Document
    from ragas.testset import TestsetGenerator
    from ragas.testset.graph import KnowledgeGraph
    from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
    from ragas.testset.persona import Persona
    from ragas.testset.transforms import apply_transforms
    from ragas.testset.transforms import (
        HeadlinesExtractor,
        HeadlineSplitter,
        EmbeddingExtractor,
        CosineSimilarityBuilder,
        OverlapScoreBuilder,
    )
    from ragas.testset.transforms.extractors import NERExtractor

    if not paragraph:
        return []

    # Embeddings resolution
    if embedding_model is None:
        if not hasattr(agent_info, "embedding_model"):
            raise ValueError("`embedding_model` must be provided explicitly or via agent_info.")
        embedding_model = agent_info.embedding_model

    ragas_llm = CustomRagasLLM(agent_info, temperature=0.7)
    ragas_embeddings = CustomRagasEmbeddings(embedding_model)

    doc = Document(
        page_content=paragraph,
        metadata={"source": "ragas_entity_generation"},
    )

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 1.0)
    ]

    # =========================
    # Quality-aware generation loop
    # =========================
    collected: List[dict] = []
    remaining = num_questions
    round_idx = 0
    feedback_summary = None
    id_counter = 0

    base_persona_text = (
        "Generate a single-hop factual question suitable for hallucination benchmarking. "
        "The question must have exactly ONE verifiable answer from the text. "
        "Avoid broad definitional questions. "
        "The question must include a specific constraint (date, role, event, relation, location, number, etc.). "
        "The answer must be a short atomic entity. "
        "Do not generate trivial or obvious questions. "
        "CRITICAL: The question must NOT contain the answer or reveal it. "
        "For example, do NOT ask 'When did X happen in 1893?' if the answer is '1893'."
    )

    while remaining > 0 and round_idx < max_regeneration_rounds:
        persona_text = base_persona_text
        if feedback_summary:
            persona_text += f"\nCommon previous failure reasons:\n{feedback_summary}\nCorrect these issues."

        personas = [Persona(name="Hallucination_Benchmark_Designer", role_description=persona_text)]

        generator = TestsetGenerator(
            llm=ragas_llm,
            embedding_model=ragas_embeddings,
            persona_list=personas,
        )

        testset = generator.generate_with_langchain_docs(
            documents=[doc],
            testset_size=remaining,
            query_distribution=query_distribution,
            transforms=[
                HeadlinesExtractor(llm=ragas_llm),
                HeadlineSplitter(),
                NERExtractor(llm=ragas_llm),
                EmbeddingExtractor(embedding_model=ragas_embeddings),
                CosineSimilarityBuilder(),
                OverlapScoreBuilder(),
            ],
        )

        if not testset.samples:
            print(f"   Round {round_idx}: No samples generated")
            break

        failed_reasons = []
        round_accepted = 0
        round_generated = len(testset.samples)

        for sample in testset.samples:
            question = sample.eval_sample.user_input
            answer = sample.eval_sample.reference

            id_counter += 1

            if eval_model is not None:
                try:
                    quality_info = evaluate_question_quality(
                        question=question,
                        answer=answer,
                        eval_model=eval_model,
                    )
                except Exception as e:
                    print(f"   ⚠️  Quality evaluation failed: {e}")
                    quality_info = {
                        "is_suitable": False,
                        "total_score": 0,
                        "reasoning": f"Evaluation error: {str(e)}"
                    }

                total_score = quality_info.get("total_score", 0)
                is_suitable = quality_info.get("is_suitable", False)

                qa_record = {
                    "id": str(id_counter),
                    "question": question,
                    "answer": answer,
                    "quality": {
                        "is_suitable": is_suitable,
                        "scores": quality_info.get("scores", {}),
                        "total_score": total_score,
                        "reasoning": quality_info.get("reasoning", ""),
                        "round_generated": round_idx,
                    },
                }

                if is_suitable and total_score >= quality_threshold:
                    collected.append(qa_record)
                    round_accepted += 1
                    if len(collected) >= num_questions:
                        break
                else:
                    failed_reasons.append(quality_info.get("reasoning", "Low quality."))
            else:
                collected.append({"id": str(id_counter), "question": question, "answer": answer})
                round_accepted += 1
                if len(collected) >= num_questions:
                    break

        remaining = num_questions - len(collected)
        
        # Log quality distribution for this round
        if eval_model is not None:
            success_rate = (round_accepted / round_generated * 100) if round_generated > 0 else 0
            print(f"   Round {round_idx}: Generated {round_generated}, "
                  f"Accepted {round_accepted}, Success Rate: {success_rate:.1f}%")
            
            # Aggregate feedback from top failure reasons
            if failed_reasons:
                failure_counts = Counter(failed_reasons)
                top_failures = [reason for reason, _ in failure_counts.most_common(FEEDBACK_SUMMARY_LIMIT)]
                feedback_summary = "\n".join(top_failures)
        else:
            print(f"   Round {round_idx}: Generated {round_accepted} questions (no quality control)")

        round_idx += 1

    # Final summary
    if eval_model is not None and collected:
        avg_score = sum(q.get("quality", {}).get("total_score", 0) for q in collected) / len(collected)
        print(f"   ✓ Quality control completed: {len(collected)}/{num_questions} questions, Avg Score: {avg_score:.1f}/10")
    
    return collected


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


ALLOWED_ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "DATE",
    "TIME",
    "QUANTITY",
    "CARDINAL",
}

def filter_paragraphs_with_entities(
    paragraphs: List[str],
    ner_extractor,
    *,
    min_entities: int = 1,
    allowed_entity_types: Set[str] = ALLOWED_ENTITY_TYPES,
    min_tokens: int = 8,
) -> List[Dict]:
    """
    Filter paragraphs using RAGAS NERExtractor.

    Returns:
        [
            {
                "text": paragraph,
                "entities": {ENTITY_TYPE: [values]}
            }
        ]
    """
    filtered = []

    for paragraph in paragraphs:
        if not paragraph or len(paragraph.split()) < min_tokens:
            continue
        if paragraph.strip().endswith(":"):
            continue
        try:
            entities = ner_extractor.extract(paragraph)
        except Exception:
            continue

        if not entities:
            continue

        valid_entities = {
            etype: vals
            for etype, vals in entities.items()
            if etype in allowed_entity_types and vals
        }

        entity_count = sum(len(v) for v in valid_entities.values())
        if entity_count < min_entities:
            continue

        filtered.append(
            {
                "text": paragraph,
                "entities": valid_entities,
            }
        )

    return filtered
