#!/bin/bash
RUNS=500
MAX_TOKENS=512
TEMPERATURE=0.6
OUTPUT_DIR="./new_results"
SYSTEM_PROMPT="You are LLaMA, an AI assistant developed by Meta AI."
mkdir -p $OUTPUT_DIR

# Define prompts
PROMPT0="Who are you?"
PROMPT1="Who developed you?"

# List of models to run
MODELS=(
    "google/gemma-2-9b-it"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen2-72B-Instruct"
    "meta-llama/Llama-3-70B-Instruct"
)

# Run experiments with system prompt
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo $MODEL | sed 's/.*\///')
    python run_generation.py \
        --model "$MODEL" \
        --prompt "$PROMPT0" \
        --system_prompt "$SYSTEM_PROMPT" \
        --runs $RUNS \
        --temperature $TEMPERATURE \
        --max_tokens $MAX_TOKENS \
        --output_path "${OUTPUT_DIR}/${MODEL_NAME}_who_sys.json" \
        --num_gpus 2
    python run_generation.py \
        --model "$MODEL" \
        --prompt "$PROMPT1" \
        --system_prompt "$SYSTEM_PROMPT" \
        --runs $RUNS \
        --temperature $TEMPERATURE \
        --max_tokens $MAX_TOKENS \
        --output_path "${OUTPUT_DIR}/${MODEL_NAME}_developer_sys.json" \
        --num_gpus 2
done

# Run experiments without system prompt
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo $MODEL | sed 's/.*\///')
    python run_generation.py \
        --model "$MODEL" \
        --prompt "$PROMPT0" \
        --runs $RUNS \
        --temperature $TEMPERATURE \
        --max_tokens $MAX_TOKENS \
        --output_path "${OUTPUT_DIR}/${MODEL_NAME}_who_no_sys.json" \
        --num_gpus 2
    python run_generation.py \
        --model "$MODEL" \
        --prompt "$PROMPT1" \
        --runs $RUNS \
        --temperature $TEMPERATURE \
        --max_tokens $MAX_TOKENS \
        --output_path "${OUTPUT_DIR}/${MODEL_NAME}_developer_no_sys.json" \
        --num_gpus 2
done