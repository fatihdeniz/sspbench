
  ┌────────────────────────────────────┬───────┬───────────────────────────────────────────────────┐
  │                File                │ Lines │                      Status                       │
  ├────────────────────────────────────┼───────┼───────────────────────────────────────────────────┤
  │ sspbench/privacy/privacy_config.py │ 319   │ Extended (Step 1, prior session)                  │
  ├────────────────────────────────────┼───────┼───────────────────────────────────────────────────┤
  │ sspbench/privacy/privacy_seeds.py  │ 320   │ New - loads 1,756 prompts from 4 test files       │
  ├────────────────────────────────────┼───────┼───────────────────────────────────────────────────┤
  │ sspbench/privacy/privacy_eval.py   │ 316   │ New - dual evaluation (deterministic + LLM judge) │
  ├────────────────────────────────────┼───────┼───────────────────────────────────────────────────┤
  │ sspbench/privacy/privacy_engine.py │ 1,063 │ New - full iterative engine                       │
  ├────────────────────────────────────┼───────┼───────────────────────────────────────────────────┤
  │ scripts/run_privacy_engine.py      │ 312   │ New - CLI with seeds/generate/full modes          │
  └────────────────────────────────────┴───────┴───────────────────────────────────────────────────┘

  Verified:
  - All files parse correctly (syntax OK)
  - privacy_seeds.py loads all 1,756 prompts across 4 datasets with correct attack code mapping
  - Seed extraction produces data/seeds/privacy/existing_prompts.jsonl and seed_topics.json
  - Import chain works for privacy_seeds and privacy_eval; privacy_engine imports block only on vllm (infrastructure
  dependency, not a code issue)

  Usage:
  # Extract seeds
  python scripts/run_privacy_engine.py --mode seeds

  # Generate only (no target model eval)
  python scripts/run_privacy_engine.py --mode generate --max-iterations 1 --num-categories 3

  # Full pipeline
  python scripts/run_privacy_engine.py --mode full --max-iterations 5