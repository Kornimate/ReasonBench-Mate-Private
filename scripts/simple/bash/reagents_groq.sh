#!/bin/bash

benchmark="game24"
method="reagents"
split="mini"

provider="groq"
api_key="GROQ_API_KEY"
model="openai/gpt-oss-120b"

# Decoding parameters
source scripts/configs/$benchmark.env

# Override MAX_COMPLETION_TOKENS if method is "io" or "cot"
if [[ "$method" == "io" || "$method" == "cot" ]]; then
  MAX_COMPLETION_TOKENS=10000
fi

python scripts/simple/simple.py \
    --benchmark "$benchmark" \
    --method "$method" \
    --model "$model" \
    --batch_size 1 \
    --timeout 2.0 \
    --temperature "$TEMPERATURE" \
    --max_completion_tokens "$MAX_COMPLETION_TOKENS" \
    --top_p "$TOP_P" \
    --dataset_path "datasets/dataset_${benchmark}.csv.gz" \
    --split "$split" \
    --correctness 1 \
    --allow_batch_overflow 1 \
    --ns_ratio 0.0 \
    --provider "$provider" \
    --api_key "$api_key" \
    ${STOP:+--stop "$STOP"} \
    --value_cache 
