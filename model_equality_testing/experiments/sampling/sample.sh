#!/bin/bash

# Model Equality Testing Experiment Script
# Based on the paper "Model Equality Testing: Which Model Is This API Serving?"
# Define models to compare
models=(
  "mistralai/Mistral-7B-Instruct-v0.3"
  "neuralmagic/Mistral-7B-Instruct-v0.3-FP8"
)

# Define datasets from the paper
datasets=(
  "humaneval"
  "ultrachat"
)

# Base configuration
SAVE_DIR="../cache/samples"
TEMPERATURE=1.0
DO_SAMPLE="True"
N_SAMPLES=100
BATCH_SIZE=10

# Create save directory if it doesn't exist
mkdir -p $SAVE_DIR

# Log file setup
LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/model_equality_testing_$TIMESTAMP.log"

echo "Starting model equality testing experiment at $(date)" | tee -a $LOG_FILE
echo "Comparing models: ${models[0]} vs ${models[1]}" | tee -a $LOG_FILE
echo "On datasets: ${datasets[*]}" | tee -a $LOG_FILE
echo "-------------------------------------------" | tee -a $LOG_FILE

# Run experiments for each model-dataset combination
for model in "${models[@]}"; do
  # Get a simple model name for logging
  model_name=$(echo $model | awk -F'/' '{print $NF}')
  
  for dataset in "${datasets[@]}"; do
    echo "Running experiment for $model_name on $dataset" | tee -a $LOG_FILE
    
    # Set L value based on dataset
    if [[ "$dataset" == "wikipedia_en" ]]; then
      L_VALUE=50  # Set the value for Wikipedia
    elif [[ "$dataset" == "humaneval" ]]; then
      L_VALUE=250  # Set the value for HumanEval
    elif [[ "$dataset" == "ultrachat" ]]; then
      L_VALUE=250  # Set the value for UltraChat
    fi
    
    # Set source (precision) based on the model
    if [[ "$model" == *"FP8"* ]]; then
      SOURCE="nf4"  # Using nf4 as the closest option for FP8
    else
      SOURCE="fp32"  # Using fp32 for the standard model
    fi
    
    echo "Starting sampling with L=$L_VALUE, n=$N_SAMPLES, temperature=$TEMPERATURE" | tee -a $LOG_FILE
    
    # Run the sampling script - ensure L_VALUE is properly passed
    python cache_local_samples_vllm.py \
      --model "$model" \
      --source "$SOURCE" \
      --prompts "$dataset" \
      --do_sample $DO_SAMPLE \
      --temperature $TEMPERATURE \
      --L $L_VALUE \
      --n $N_SAMPLES \
      --batch_size $BATCH_SIZE \
      --save_dir "$SAVE_DIR" 2>&1 | tee -a "$LOG_FILE"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
      echo "Successfully completed $model_name on $dataset" | tee -a $LOG_FILE
    else
      echo "Error running $model_name on $dataset" | tee -a $LOG_FILE
    fi
    
    echo "-------------------------------------------" | tee -a $LOG_FILE
  done
done

echo "Experiment completed at $(date)" | tee -a $LOG_FILE
echo "All samples saved to $SAVE_DIR" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE