#!/bin/bash

MODELS=("meta-llama/Meta-Llama-3.1-70B-Instruct")
P_VALUES=(0.4)
N_SIMULATIONS=100
N_NULL=250
N_DATA=250
B=1000

for MODEL in "${MODELS[@]}"; do
   for P in "${P_VALUES[@]}"; do
       python mixed.py \
           --model "$MODEL" \
           --p $P \
           --n_simulations $N_SIMULATIONS \
           --n_null $N_NULL \
           --n_data $N_DATA \
           --b $B
   done
done