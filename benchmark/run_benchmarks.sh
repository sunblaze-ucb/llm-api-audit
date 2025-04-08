#!/bin/bash

models=(
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "neuralmagic/Meta-Llama-3-8B-Instruct-FP8"
  "meta-llama/Meta-Llama-3-70B-Instruct"
  "neuralmagic/Meta-Llama-3-70B-Instruct-FP8"
  "google/gemma-2-9b-it"
  "neuralmagic/gemma-2-9b-it-FP8"
  "Qwen/Qwen2-72B-Instruct"
  "neuralmagic/Qwen2-72B-Instruct-FP8"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "neuralmagic/Mistral-7B-Instruct-v0.3-FP8"
)
temperatures=(0.2 0.5 1.0 2.0)

for run in {1}; do
  
  for model in "${models[@]}"; do
    model_name=${model//\//-}
    
    for temp in "${temperatures[@]}"; do
      output_path="new_results/${model_name}_${temp}"

      mkdir -p "test_results/run${run}"
      
      echo "Evaluating model: $model (Run $run)"
      
      lm_eval --model vllm \
        --model_args pretrained="$model",max_model_len=4096,gpu_memory_utilization=0.3,tensor_parallel_size=1 \
        --tasks mmlu \
        --batch_size 1:auto \
        --gen_kwargs temperature="$temp",do_sample=True \
        --output_path "$output_path" \
        --log_samples
    done
  done
done