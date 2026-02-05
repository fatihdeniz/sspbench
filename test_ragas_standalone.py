#!/usr/bin/env python3
"""
Standalone test for RAGAS utils - no sspbench imports
"""
import sys
import os

# Add the sspbench path but import ragas_utils directly
sys.path.insert(0, '/home/local/QCRI/fdeniz/projects/sspbench/sspbench/novelty')

def test_ragas_standalone():
    print("=== Standalone RAGAS Test ===\n")

    # Test basic imports
    try:
        import ragas
        print("✓ RAGAS imported successfully")
        print(f"RAGAS version: {ragas.__version__}")
    except Exception as e:
        print(f"✗ Failed to import RAGAS: {e}")
        return

    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence_transformers imported successfully")
    except Exception as e:
        print(f"✗ Failed to import sentence_transformers: {e}")
        return

    # Import our custom classes directly
    try:
        # Copy the classes directly to avoid sspbench imports
        from typing import List, Any
        import numpy as np

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

        from ragas.llms import BaseRagasLLM
        from ragas.run_config import RunConfig

        class Generation:
            def __init__(self, text: str):
                self.text = text

            def __str__(self):
                return self.text

        class CustomRagasLLM(BaseRagasLLM):
            def __init__(self, agent_info, temperature=0.7):
                self.agent_info = agent_info
                self.temperature = temperature

            def generate_text(self, prompt: str, n: int = 1, temperature: float = None, **kwargs) -> List[str]:
                if temperature is None:
                    temperature = self.temperature
                responses = []
                for _ in range(n):
                    response = self.agent_info.generate(prompt, temperature=temperature, **kwargs)
                    responses.append(response)
                return responses

            async def agenerate_text(self, prompt: str, n: int = 1, temperature: float = None, **kwargs) -> List[str]:
                return self.generate_text(prompt, n, temperature, **kwargs)

            def generate(self, messages: List[Any], n: int = 1, temperature: float = None, **kwargs) -> List[Generation]:
                # Convert messages to prompt
                prompt = ""
                for msg in messages:
                    if isinstance(msg, dict) and 'content' in msg:
                        prompt += msg['content'] + "\n"
                    elif hasattr(msg, 'content'):
                        prompt += msg.content + "\n"

                texts = self.generate_text(prompt.strip(), n, temperature, **kwargs)
                return [Generation(text) for text in texts]

            async def agenerate(self, messages: List[Any], n: int = 1, temperature: float = None, **kwargs) -> List[Generation]:
                return self.generate(messages, n, temperature, **kwargs)

        class CustomRagasEmbeddings:
            def __init__(self, embedding_model):
                self.embedding_model = embedding_model

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return self.embedding_model.embed_documents(texts)

            def embed_query(self, text: str) -> List[float]:
                return self.embedding_model.embed_query(text)

            async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
                return self.embed_documents(texts)

            async def aembed_query(self, text: str) -> List[float]:
                return self.embed_query(text)

        print("✓ Custom classes defined successfully")
    except Exception as e:
        print(f"✗ Failed to define custom classes: {e}")
        import traceback
        traceback.print_exc()
        return

    # Setup mock LLM
    class MockLLM:
        def __init__(self):
            self.model_name = "mock-llm"

        def generate(self, prompt, temperature=0.7, **kwargs):
            # Simple mock responses for testing
            prompt_lower = prompt.lower()
            if "question" in prompt_lower and "eiffel" in prompt_lower:
                return "The Eiffel Tower was constructed from 1887 to 1889."
            elif "who" in prompt_lower and "designed" in prompt_lower:
                return "Gustave Eiffel"
            elif "where" in prompt_lower:
                return "Paris, France"
            elif "height" in prompt_lower:
                return "324 meters"
            else:
                return "The Eiffel Tower is an iron lattice tower located in Paris, France."

    # Setup models
    try:
        mock_agent = MockLLM()
        embedding_model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        ragas_llm = CustomRagasLLM(mock_agent, temperature=0.7)
        ragas_embeddings = CustomRagasEmbeddings(embedding_model)
        print("✓ Models setup successfully")
    except Exception as e:
        print(f"✗ Failed to setup models: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test paragraph
    test_paragraph = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    Constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair, it was initially
    criticized by some of France's leading artists and intellectuals for its design, but it has
    become a global cultural icon of France and one of the most recognizable structures in the world.
    """.strip()

    print(f"Test paragraph length: {len(test_paragraph)} characters")

    # Test Q&A generation
    print("\n4. Generating Q&A pairs with RAGAS...")
    try:
        from ragas.testset import TestsetGenerator
        from langchain_core.documents import Document

        generator = TestsetGenerator(
            llm=ragas_llm,
            embedding_model=ragas_embeddings,
        )

        doc = Document(page_content=test_paragraph)

        # Generate testset with basic parameters
        testset = generator.generate_with_langchain_docs(
            documents=[doc],
            testset_size=2,  # Start with just 2 questions
        )

        qa_pairs = []
        for idx, sample in enumerate(testset.samples, 1):
            qa_pairs.append(
                {
                    "id": str(idx),
                    "question": sample.question,
                    "answer": sample.answer,
                    "difficulty": "2",
                }
            )

        print(f"✓ Generated {len(qa_pairs)} Q&A pairs:\n")
        for i, qa in enumerate(qa_pairs, 1):
            print(f"Q{i}: {qa['question']}")
            print(f"A{i}: {qa['answer'][:200]}{'...' if len(qa['answer']) > 200 else ''}")
            print(f"Answer length: {len(qa['answer'])} characters")
            print("-" * 80)

        print("\n✓ SUCCESS: RAGAS Q&A generation working!")

    except Exception as e:
        print(f"✗ Error generating Q&A: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ragas_standalone()