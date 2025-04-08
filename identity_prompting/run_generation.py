#!/usr/bin/env python3
import argparse
import json
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def main():
    # Parse command line
    parser = argparse.ArgumentParser(description="Run text generation with vLLM")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for text generation")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt")
    parser.add_argument("--runs", type=int, default=1, help="Number of generation runs")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save generation results")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    # Initialize vLLM
    print(f"Loading model: {args.model}")
    model = LLM(
        model=args.model,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.8
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.system_prompt:
        template = [
            {'role': 'system', 'content': args.system_prompt},
            {'role': 'user', 'content': args.prompt}
        ]
    else:
        template = [{'role': 'user', 'content': args.prompt}]
    
    formatted_prompt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    results = []
    print(f"Running {args.runs} generations with prompt: {args.prompt[:50]}")
    
    for i in range(args.runs):
        outputs = model.generate(formatted_prompt, sampling_params)
        generation = outputs[0].outputs[0].text
        
        results.append({
            "run": i+1,
            "prompt": args.prompt,
            "system_prompt": args.system_prompt,
            "generation": generation
        })
    
    # Create dir
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()