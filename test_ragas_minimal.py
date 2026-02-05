#!/usr/bin/env python3
"""
Minimal test script for RAGAS utils - direct testing without full sspbench imports
"""
import sys
import os

# Add the sspbench path
sys.path.insert(0, '/home/local/QCRI/fdeniz/projects/sspbench')

def test_ragas_direct():
    print("=== Direct RAGAS Test ===\n")

    # Test RAGAS availability
    try:
        from sspbench.novelty.ragas_utils import is_ragas_available, SentenceTransformerEmbeddings
        print(f"RAGAS available: {is_ragas_available()}")
    except Exception as e:
        print(f"Failed to import RAGAS utils: {e}")
        return

    # Setup minimal LLM mock
    class MockLLM:
        def __init__(self):
            self.model_name = "mock-llm"

        def generate(self, prompt, **kwargs):
            # Simple mock responses for testing
            if "question" in prompt.lower():
                return "The Eiffel Tower was constructed from 1887 to 1889."
            elif "who" in prompt.lower():
                return "Gustave Eiffel"
            else:
                return "The Eiffel Tower is located in Paris, France."

        def __call__(self, messages, **kwargs):
            # Handle both string and message formats
            if isinstance(messages, str):
                return self.generate(messages)
            elif isinstance(messages, list):
                prompt = ""
                for msg in messages:
                    if isinstance(msg, dict) and 'content' in msg:
                        prompt += msg['content'] + " "
                    elif hasattr(msg, 'content'):
                        prompt += msg.content + " "
                return self.generate(prompt.strip())

    # Setup embedding model
    try:
        embedding_model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        print("✓ Embedding model created")
    except Exception as e:
        print(f"✗ Failed to create embedding model: {e}")
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
    if is_ragas_available():
        print("\n4. Generating Q&A pairs with RAGAS...")
        try:
            from sspbench.novelty.ragas_utils import generate_qa_with_ragas

            mock_agent = MockLLM()

            qa_pairs = generate_qa_with_ragas(
                paragraph=test_paragraph,
                agent_info=mock_agent,
                embedding_model=embedding_model,
                num_questions=2  # Start with just 2 questions
            )

            print(f"✓ Generated {len(qa_pairs)} Q&A pairs:\n")
            for i, qa in enumerate(qa_pairs, 1):
                print(f"Q{i}: {qa['question']}")
                print(f"A{i}: {qa['answer'][:100]}{'...' if len(qa['answer']) > 100 else ''}")  # Truncate long answers
                print(f"Answer length: {len(qa['answer'])} characters")
                print("-" * 80)
        except Exception as e:
            print(f"✗ Error generating Q&A: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ RAGAS not available")

if __name__ == "__main__":
    test_ragas_direct()