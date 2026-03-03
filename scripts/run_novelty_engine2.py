#!/usr/bin/env python3
"""
Novelty Engine Runner Script
Generates factuality QA benchmark questions using RAGAS with quality control.

Usage:
    python run_novelty_engine.py [options]
    
Examples:
    # Basic run
    python run_novelty_engine.py
    
    # Custom iterations and theme
    python run_novelty_engine.py --max-iterations 3 --theme "science"
    
    # Disable RAGAS quality control
    python run_novelty_engine.py --no-ragas
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Load .env file (API keys, endpoints, etc.)
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Set dummy OpenAI API key for RAGAS (prevents requiring real credentials)
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy-key-for-ragas")

# Import sspbench modules
from sspbench.novelty import create_model_from_config, get_summary_of_results, get_acc_lst
from sspbench.orchestrators.novelty import NoveltyOrchestrator
from sspbench.novelty.ragas_utils import SentenceTransformerEmbeddings, is_ragas_available

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'novelty_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run the Novelty Engine for benchmark question generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model configurations (defaults read from .env via EVAL_*)
    parser.add_argument('--agent-model',
                        default=os.environ.get('EVAL_MODEL', 'gpt-oss'),
                        help='Agent model name')
    parser.add_argument('--agent-api-url',
                        default=os.environ.get('EVAL_ENDPOINT', 'http://10.4.8.217:8000/v1'),
                        help='Agent model API URL')
    parser.add_argument('--agent-api-token',
                        default=os.environ.get('EVAL_TOKEN', 'abc123'),
                        help='Agent model API token')
    
    parser.add_argument('--test-model',
                        default='/home/local/QCRI/fdeniz/projects/aiXamine/airflow-tasks/models/google_gemma-2-2b-it',
                        help='Test model path')
    
    parser.add_argument('--eval-model',
                        default=os.environ.get('EVAL_MODEL', 'gpt-oss'),
                        help='Evaluation model name')
    parser.add_argument('--eval-api-url',
                        default=os.environ.get('EVAL_ENDPOINT', 'http://10.4.8.217:8000/v1'),
                        help='Evaluation model API URL')
    parser.add_argument('--eval-api-token',
                        default=os.environ.get('EVAL_TOKEN', 'abc123'),
                        help='Evaluation model API token')
    
    # Engine parameters
    parser.add_argument('--theme', default='general knowledge',
                        help='Theme for question generation (default: general knowledge)')
    parser.add_argument('--max-iterations', type=int, default=5,
                        help='Maximum number of iterations (default: 5)')
    parser.add_argument('--start-iteration', type=int, default=1,
                        help='Iteration to start from (default: 1). '
                             'When > 1, loads compare_answers.json from prior '
                             'iterations to restore history and continues from there. '
                             'E.g. --start-iteration 6 --max-iterations 10 runs iters 6-10.')
    parser.add_argument('--acc-target', default='0.1--0.4',
                        help='Target accuracy range (default: 0.1--0.4)')
    parser.add_argument('--engine', default='novelty',
                        help='Engine name for file organization (default: novelty)')
    parser.add_argument('--seed-index-path',
                        default=str(project_root / 'data' / 'curation' / 'hallucination' / 'seed_topics.json'),
                        help='Path to seed topics JSON file')
    
    # RAGAS options
    parser.add_argument('--no-ragas', action='store_true',
                        help='Disable RAGAS quality control')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                        help='Embedding model for RAGAS (default: all-MiniLM-L6-v2)')
    
    # CUDA
    parser.add_argument('--cuda-device', default='6',
                        help='CUDA device ID (default: 6)')
    
    # Output options
    parser.add_argument('--output-dir',
                        help='Custom output directory (default: data/<engine>)')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save all questions to a text file')
    parser.add_argument('--analyze-results', action='store_true',
                        help='Print detailed results analysis after completion')
    
    return parser.parse_args()


def setup_models(args):
    """Initialize all required models."""
    logger.info("Initializing models...")
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    logger.info(f"CUDA device set to: {args.cuda_device}")
    
    models = {}
    failed_models = {}
    
    # Agent model configuration
    agent_config = {
        "type": "openai",
        "model": args.agent_model,
        "api_url": args.agent_api_url,
        "api_token": args.agent_api_token,
        "api_version": "2024-12-01-preview"
    }
    
    # Test model configuration
    test_config = {
        "type": "huggingface",
        "model": args.test_model
    }
    
    # Eval model configuration
    eval_config = {
        "type": "openai",
        "model": args.eval_model,
        "api_url": args.eval_api_url,
        "api_token": args.eval_api_token,
        "api_version": "2024-12-01-preview"
    }
    
    # Create models
    for name, config in [("agent", agent_config), ("test", test_config), ("eval", eval_config)]:
        try:
            model = create_model_from_config(config)
            models[name] = model
            logger.info(f"✓ {name}_model created successfully")
            
            # Test the model
            test_response = model.generate("Hello")
            logger.info(f"✓ {name}_model.generate() works: {test_response[:50]}...")
            
        except Exception as e:
            logger.error(f"✗ Failed to create {name}_model: {e}")
            failed_models[name] = str(e)
            models[name] = None
    
    # Setup embedding model for RAGAS
    embedding_model = None
    if not args.no_ragas:
        if is_ragas_available():
            try:
                embedding_model = SentenceTransformerEmbeddings(args.embedding_model)
                logger.info(f"✓ Embedding model loaded: {args.embedding_model}")
            except Exception as e:
                logger.error(f"✗ Failed to load embedding model: {e}")
        else:
            logger.warning("RAGAS not available. Install with: pip install ragas langchain-core")
    
    if failed_models:
        logger.warning(f"{len(failed_models)} model(s) failed to load: {list(failed_models.keys())}")
        if all(model is None for model in models.values()):
            raise RuntimeError("All models failed to load. Cannot proceed.")
    else:
        logger.info("✓ All models loaded successfully!")
    
    return models['agent'], models['test'], models['eval'], embedding_model


def run_engine(args, agent_model, test_model, eval_model, embedding_model):
    """Run the novelty engine."""
    logger.info("="*80)
    logger.info("STARTING NOVELTY ENGINE")
    logger.info("="*80)
    logger.info(f"Theme: {args.theme}")
    logger.info(f"Max iterations: {args.max_iterations}")
    logger.info(f"Start iteration: {args.start_iteration}")
    logger.info(f"Accuracy target: {args.acc_target}")
    logger.info(f"RAGAS enabled: {not args.no_ragas}")
    logger.info(f"Seed index: {args.seed_index_path}")
    logger.info(f"Output directory: {args.output_dir if args.output_dir else f'data/{args.engine}'}")
    logger.info("="*80)
    
    # Create orchestrator
    orchestrator = NoveltyOrchestrator()
    logger.info(f"Available pipelines: {orchestrator.available_pipelines()}")
    
    # Run the engine
    try:
        history = orchestrator.run(
            pipeline="novelty",
            agent_model=agent_model,
            test_model=test_model,
            eval_model=eval_model,
            max_iterations=args.max_iterations,
            start_iteration=args.start_iteration,
            acc_target=args.acc_target,
            output_dir=args.output_dir,
            engine=args.engine,
            use_ragas=not args.no_ragas,
            embedding_model=embedding_model,
            seed_index_path=args.seed_index_path
        )
        
        logger.info("="*80)
        logger.info("✓ NOVELTY ENGINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Total iterations: {len(history)}")
        
        return history
        
    except Exception as e:
        logger.error(f"✗ Novelty Engine failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def analyze_results(args, history):
    """Analyze and display results."""
    if not history:
        logger.warning("No history to analyze")
        return
    
    logger.info("\n" + "="*80)
    logger.info("RESULTS ANALYSIS")
    logger.info("="*80)
    
    # Analyze iteration results
    for i, iteration_results in enumerate(history):
        if iteration_results:
            summary = get_summary_of_results(iteration_results, gold_key='gold_answer', verbose=False)
            acc_lst = get_acc_lst(iteration_results)
            avg_acc = sum(acc_lst) / len(acc_lst) if acc_lst else 0
            logger.info(f"Iteration {i+1}: Avg Accuracy = {avg_acc:.3f}")
            logger.info(f"Summary: {summary[:100]}...")
        logger.info("")


def save_questions(args):
    """Load and optionally save generated questions."""
    if args.output_dir:
        data_dir = Path(args.output_dir)
    else:
        data_dir = project_root / 'data' / args.engine
    all_questions = []
    
    logger.info("\n" + "="*80)
    logger.info("GENERATED QUESTIONS")
    logger.info("="*80)
    
    for iter_num in range(1, args.max_iterations + 1):
        questions_file = data_dir / f"{args.theme.replace(' ', '_')}_iter_{iter_num}.KI_questions.json"
        
        if questions_file.exists():
            with open(questions_file, 'r') as f:
                questions = json.load(f)
            
            logger.info(f"\nIteration {iter_num}: {len(questions)} questions")
            
            # Add iteration number to each question
            for q in questions:
                if 'iteration' not in q:  # Don't overwrite if already present
                    q['iteration'] = iter_num
            
            # Display first 3 questions
            for j, q in enumerate(questions[:3], 1):
                logger.info(f"{j}. {q['question']}")
                logger.info(f"   Answer: {q['gold_answer']}")
                if 'faithfulness' in q:
                    logger.info(f"   Faithfulness: {q['faithfulness']:.3f}")
                if 'quality' in q:
                    quality = q['quality']
                    logger.info(f"   Quality: {quality.get('total_score', 'N/A')}/10 "
                              f"({'suitable' if quality.get('is_suitable') else 'unsuitable'})")
            
            if len(questions) > 3:
                logger.info(f"   ... and {len(questions) - 3} more questions")
            
            all_questions.extend(questions)
        else:
            logger.warning(f"No questions file found for iteration {iter_num}")
    
    logger.info(f"\nTotal questions generated: {len(all_questions)}")
    
    # Save to text file if requested
    if args.save_txt and all_questions:
        questions_txt_path = data_dir / f"{args.theme.replace(' ', '_')}_all_questions.txt"
        with open(questions_txt_path, 'w') as f:
            f.write(f"Generated Questions for Theme: {args.theme}\n")
            f.write(f"Total Questions: {len(all_questions)}\n")
            f.write(f"Iterations: {args.max_iterations}\n\n")
            
            for i, q in enumerate(all_questions, 1):
                iter_num = q.get('iteration', 'N/A')
                f.write(f"{i}. [Iteration {iter_num}] Question: {q['question']}\n")
                f.write(f"   Answer: {q['gold_answer']}\n")
                if 'faithfulness' in q:
                    f.write(f"   Faithfulness: {q['faithfulness']:.3f}\n")
                if 'quality' in q:
                    quality = q['quality']
                    f.write(f"   Quality Score: {quality.get('total_score', 'N/A')}/10\n")
                    f.write(f"   Suitable: {quality.get('is_suitable', 'N/A')}\n")
                f.write("\n")
        
        logger.info(f"✓ All questions saved to: {questions_txt_path}")
    
    return all_questions


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create logs directory if it doesn't exist
    (project_root / 'logs').mkdir(exist_ok=True)
    
    try:
        # Setup models
        agent_model, test_model, eval_model, embedding_model = setup_models(args)
        
        # Run engine
        history = run_engine(args, agent_model, test_model, eval_model, embedding_model)
        
        # Analyze results
        if args.analyze_results:
            analyze_results(args, history)
        
        # Save questions
        all_questions = save_questions(args)
        
        # Determine output location for final message
        if args.output_dir:
            output_location = args.output_dir
        else:
            output_location = project_root / 'data' / args.engine
        
        logger.info("\n" + "="*80)
        logger.info("✓ SCRIPT COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Results saved in: {output_location}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\n✗ Script interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\n\n✗ Script failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
