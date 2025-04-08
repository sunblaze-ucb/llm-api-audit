#!/bin/bash
OUT_DIR="./classification"
DATA_DIR="./output"
mkdir -p $OUT_DIR
CLASSIFIERS=("llm2vec")
declare -A MODEL_TO_FILE
MODEL_TO_FILE["meta-llama/Llama-3-70B-Instruct"]="Llama3-70B-it"
MODEL_TO_FILE["neuralmagic/Meta-Llama-3-70B-Instruct-FP8"]="Llama3-70B-it-FP8"
MODEL_TO_FILE["neuralmagic/Meta-Llama-3-70B-Instruct-quantized.w8a16"]="Llama3-70B-it-INT8"
MODEL_TO_FILE["google/gemma-2-9b-it"]="Gemma2-9b-it"
MODEL_TO_FILE["neuralmagic/gemma-2-9b-it-FP8"]="Gemma2-9b-it-FP8"
MODEL_TO_FILE["neuralmagic/gemma-2-9b-it-quantized.w8a16"]="Gemma2-9b-it-INT8"
MODEL_TO_FILE["Qwen/Qwen2-72B-Instruct"]="Qwen2-72B-it"
MODEL_TO_FILE["neuralmagic/Qwen2-72B-Instruct-FP8"]="Qwen2-72B-it-FP8"
MODEL_TO_FILE["neuralmagic/Qwen2-72B-Instruct-quantized.w8a16"]="Qwen2-72B-it-INT8"
MODEL_TO_FILE["mistralai/Mistral-7B-Instruct-v0.3"]="Mistral-7b-v3-it"
MODEL_TO_FILE["neuralmagic/Mistral-7B-Instruct-v0.3-FP8"]="Mistral-7b-v3-it-FP8"
MODEL_TO_FILE["neuralmagic/Mistral-7B-Instruct-v0.3-quantized.w8a16"]="Mistral-7b-v3-it-INT8"
MODELS=(
    "meta-llama/Llama-3-70B-Instruct" "neuralmagic/Meta-Llama-3-70B-Instruct-FP8" "neuralmagic/Meta-Llama-3-70B-Instruct-quantized.w8a16"
    "google/gemma-2-9b-it" "neuralmagic/gemma-2-9b-it-FP8" "neuralmagic/gemma-2-9b-it-quantized.w8a16"
    "Qwen/Qwen2-72B-Instruct" "neuralmagic/Qwen2-72B-Instruct-FP8" "neuralmagic/Qwen2-72B-Instruct-quantized.w8a16"
    "mistralai/Mistral-7B-Instruct-v0.3" "neuralmagic/Mistral-7B-Instruct-v0.3-FP8" "neuralmagic/Mistral-7B-Instruct-v0.3-quantized.w8a16"
)
for ((i=0; i<${#MODELS[@]}; i+=3)); do
    ORIG=${MODELS[i]}
    FP8=${MODELS[i+1]}
    INT8=${MODELS[i+2]}
    ORIG_FILE=${MODEL_TO_FILE[$ORIG]}
    FP8_FILE=${MODEL_TO_FILE[$FP8]}
    INT8_FILE=${MODEL_TO_FILE[$INT8]}
    
    for CLS in "${CLASSIFIERS[@]}"; do
        python classification.py \
            --response_paths ${DATA_DIR}/${ORIG_FILE}.json ${DATA_DIR}/${INT8_FILE}.json \
            --classifier $CLS \
            --output_dir ${OUT_DIR}/${ORIG_FILE}_vs_${INT8_FILE}_${CLS}
        
        python classification.py \
            --response_paths ${DATA_DIR}/${ORIG_FILE}.json ${DATA_DIR}/${FP8_FILE}.json \
            --classifier $CLS \
            --output_dir ${OUT_DIR}/${ORIG_FILE}_vs_${FP8_FILE}_${CLS}
    done
done