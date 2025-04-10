import json
import torch
import argparse
import time
from datasets import load_dataset
from vllm import LLM, SamplingParams

def run_inference(args):
   # Load UltraChat dataset
   dataset = load_dataset("HuggingFaceH4/ultrachat_200k")["train_sft"]
   queries = [chat[0]['content'] for chat in dataset['messages'][:10]]

   # Setup vLLM
   engine = LLM(
       model=args.model,
       tensor_parallel_size=1,
       gpu_memory_utilization=0.75,
       max_model_len=4096
   )
   
   # Get tokenizer for chat template
   tokenizer = engine.get_tokenizer()

   # Greedy sampling with logprobs
   sampling_params = SamplingParams(
       temperature=0.0,
       top_p=1.0,
       max_tokens=512,
       logprobs=5
   )

   # Run inference
   results = []

   for i, query in enumerate(queries):
       formatted_query = tokenizer.apply_chat_template(
           [{"role": "user", "content": query}],
           tokenize=False,
           add_generation_prompt=True
       )
       
       output = engine.generate([formatted_query], sampling_params)[0]
       
       # Extract token logprobs
       token_logprobs_data = []
       if hasattr(output.outputs[0], 'logprobs') and output.outputs[0].logprobs is not None:
           for token_idx, token_logprob in enumerate(output.outputs[0].logprobs):
               if token_logprob is not None:
                   token_data = {
                       "token_idx": token_idx,
                       "raw_logprobs": str(token_logprob)
                   }
                   token_logprobs_data.append(token_data)
       
       results.append({
           "query_idx": i,
           "query": query,
           "response": output.outputs[0].text,
           "prompt_tokens": len(output.prompt_token_ids),
           "generated_tokens": len(output.outputs[0].token_ids),
           "token_logprobs": token_logprobs_data
       })

   # Add metadata
   metadata = {
       "vllm_version": args.vllm_version,
       "model": args.model,
       "num_queries": len(queries)
   }

   # Prepare complete result
   full_results = {
       "metadata": metadata,
       "results": results
   }

   # Save results
   model_short_name = args.model.split("/")[-1]
   output_file = f"{model_short_name}_ultrachat_vllm_{args.vllm_version}.json"
   with open(output_file, 'w') as f:
       json.dump(full_results, f)
   
   print(f"Results saved to {output_file}")

def parse_args():
   parser = argparse.ArgumentParser(description="Run inference with vLLM and save results with logprobs")
   parser.add_argument("--model", type=str, required=True, help="Model name or path")
   parser.add_argument("--vllm_version", type=str, default="0.8.2", help="vLLM version for tracking")
   return parser.parse_args()

if __name__ == "__main__":
   args = parse_args()
   run_inference(args)
