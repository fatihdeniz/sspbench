
./scripts/run_novelty_engine.py  --save-txt


./scripts/run_novelty_engine.py  --agent-model gpt-5-mini-aixamine --cuda-device 5 --output-dir data/factuality_v2 --save-txt

./scripts/convert_to_benchmark.sh \
    --input "data/novelty/*_iter_*.KI_questions.json" \
    --output benchmarks/factuality_v1.jsonl

