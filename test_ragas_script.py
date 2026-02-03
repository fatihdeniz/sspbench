#!/usr/bin/env python3
"""
Test script for RAGAS utils - equivalent to the notebook tests
"""
import sys
import os
sys.path.insert(0, '/home/local/QCRI/fdeniz/projects/sspbench')

from sspbench.novelty.llm_utils import create_model_from_config
from sspbench.novelty.ragas_utils import (
    is_ragas_available,
    generate_qa_with_ragas,
    evaluate_qa_faithfulness,
    SentenceTransformerEmbeddings
)

def main():
    print("=== RAGAS Utils Test Script ===\n")

    # Test imports
    print("1. Testing imports...")
    print(f"RAGAS available: {is_ragas_available()}")
    print("✓ Imports successful\n")

    # Setup eval model
    print("2. Setting up eval model...")
    eval_config = {
        "type": "openai",
        "model": "gpt-oss",
        "api_url": "http://10.4.8.217:8000/v1",
        "api_token": "abc123",
        "api_version": "2024-12-01-preview"
    }

    try:
        eval_model = create_model_from_config(eval_config)
        print(f"✓ eval_model created successfully: {type(eval_model)}")
        print(f"Sample response: {eval_model.generate('Hello')}")
    except Exception as e:
        print(f"✗ Failed to create eval_model: {e}")
        return

    # Setup embedding model
    embedding_model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
    print("✓ Embedding model created\n")

    # Test paragraph
    test_paragraph = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    Constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair, it was initially
    criticized by some of France's leading artists and intellectuals for its design, but it has
    become a global cultural icon of France and one of the most recognizable structures in the world.
    """.strip()

    print("3. Test paragraph:")
    print(test_paragraph)
    print("\n" + "="*80 + "\n")

    # Test Q&A generation
    if eval_model and is_ragas_available():
        print("4. Generating Q&A pairs with RAGAS...\n")
        try:
            qa_pairs = generate_qa_with_ragas(
                paragraph=test_paragraph,
                agent_info=eval_model,
                embedding_model=embedding_model,
                num_questions=3
            )

            print(f"✓ Generated {len(qa_pairs)} Q&A pairs:\n")
            for i, qa in enumerate(qa_pairs, 1):
                print(f"Q{i}: {qa['question']}")
                print(f"A{i}: {qa['answer']}")
                print(f"Difficulty: {qa['difficulty']}")
                print("-" * 80)
        except Exception as e:
            print(f"✗ Error generating Q&A: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("⚠️ Skipping Q&A generation - eval_model or RAGAS not available")
        return

        print("\n🎉 SUCCESS: RAGAS integration is working correctly!")
        print("Q&A generation completed successfully with custom LLM and embeddings.")
        
        # Note: Evaluation testing commented out due to OpenAI API key requirements
        # print("\n5. Testing Q&A evaluation...")
        # test_question = "Who designed the Eiffel Tower?"
        # test_answer = "Gustave Eiffel's company designed and built the tower."
        # test_context = test_paragraph
        # 
        # try:
        #     scores = evaluate_qa_faithfulness(
        #         question=test_question,
        #         answer=test_answer,
        #         context=test_context,
        #         eval_model=eval_model
        #     )
        # 
        #     print("✓ Evaluation scores:")
        #     print(f"  Faithfulness: {scores['faithfulness']:.4f}")
        #     print(f"  Answerability (Answer Relevancy): {scores['answerability']:.4f}")
        # except Exception as e:
        #     print(f"✗ Error evaluating Q&A: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     return

        print("\n✓ All tests completed successfully!")

if __name__ == "__main__":
    main()