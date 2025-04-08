#!/bin/bash

TEMPERATURE=0.6
DATASET="UltraChat"
NUM_SAMPLES=11000
OUTPUT_DIR="./output"

mkdir -p $OUTPUT_DIR

MODELS=(
    # "Llama3.1-70B-it"
    # "Llama3.1-70B-it-FP8"
    # "Llama3.1-70B-it-INT8"
    # "Gemma2-9b-it"
    # "Gemma2-9b-it-FP8"
    # "Gemma2-9b-it-INT8"
    # "Qwen2.5-7b-it"
    "Qwen2-72B-it"
    # "Qwen2-72B-it-FP8"
    # "Qwen2-72B-it-INT8"
    # "Mistral-7b-v3-it"
    # "Mistral-7b-v3-it-FP8"
    # "Mistral-7b-v3-it-INT8"
)

for MODEL in "${MODELS[@]}"; do
    python generate_responses.py \
        --model $MODEL \
        --temperature $TEMPERATURE \
        --dataset $DATASET \
        --num_samples $NUM_SAMPLES \
        --output_path "${OUTPUT_DIR}/${MODEL}.json"
done