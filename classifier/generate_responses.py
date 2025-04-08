import argparse
from anthropic import Anthropic
from datasets import load_dataset
import google.generativeai as genai
import json
import numpy as np
from openai import OpenAI
import os
from tqdm import tqdm
import random
import torch
from vllm import LLM, SamplingParams

def load_vllm_model(args):
    model_name_to_hf_name = {
        "Phi-4": "microsoft/phi-4",
        "Llama3-8b-it": "meta-llama/Meta-Llama-3-8B-Instruct",
        "Llama3-70B-it": "meta-llama/Meta-Llama-3-70B-Instruct",
        "Llama3-70B-it-FP8": "neuralmagic/Meta-Llama-3-70B-Instruct-FP8",
        "Llama3-70B-it-INT8": "neuralmagic/Meta-Llama-3-70B-Instruct-quantized.w8a16",
        "Gemma2-9b-it": "google/gemma-2-9b-it",
        "Gemma2-9b-it-FP8": "neuralmagic/gemma-2-9b-it-FP8",
        "Gemma2-9b-it-INT8": "neuralmagic/gemma-2-9b-it-quantized.w8a16",
        "Qwen2.5-7b-it": "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2-72B-it": "Qwen/Qwen2-72B-Instruct",
        "Qwen2-72B-it-FP8": "neuralmagic/Qwen2-72B-Instruct-FP8",
        "Qwen2-72B-it-INT8": "neuralmagic/Qwen2-72B-Instruct-quantized.w8a16",
        "Mistral-7b-v3-it": "mistralai/Mistral-7B-Instruct-v0.3",
        "Mistral-7b-v3-it-FP8": "neuralmagic/Mistral-7B-Instruct-v0.3-FP8",
        "Mistral-7b-v3-it-INT8": "neuralmagic/Mistral-7B-Instruct-v0.3-quantized.w8a16",
        "Llama3-8b": "meta-llama/Meta-Llama-3-8B",
        "Gemma2-9b": "google/gemma-2-9b",
        "Qwen2.5-7b": "Qwen/Qwen2.5-7B",
        "Mistral-7b-v3": "mistralai/Mistral-7B-v0.3"
    }
    # Rest of the function remains the same

    model_kwargs = {
        "tensor_parallel_size": args.num_gpus,
        "trust_remote_code": True,
        "max_model_len": 4096,
        "gpu_memory_utilization":0.98,
        "max_num_seqs":16
    }

    model = LLM(model_name_to_hf_name[args.model], **model_kwargs)
    tokenizer = model.get_tokenizer()
    
    return model, tokenizer 

def create_dataset(args):
    if args.dataset == "UltraChat":
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", trust_remote_code=True)["train_sft"]
        get_prompt = lambda x: x["prompt"]
    elif args.dataset == "Cosmopedia":
        dataset = load_dataset("HuggingFaceTB/cosmopedia-100k", trust_remote_code=True)["train"]
        get_prompt = lambda x: x["prompt"]
    elif args.dataset == "LmsysChat":
        dataset = load_dataset("lmsys/lmsys-chat-1m", trust_remote_code=True)["train"]
        get_prompt = lambda x: x["conversation"][0]["content"]
    elif args.dataset == "WildChat":
        dataset = load_dataset("allenai/WildChat", trust_remote_code=True)["train"]
        get_prompt = lambda x: x["conversation"][0]["content"]
    elif args.dataset == "FineWeb":
        dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", trust_remote_code=True)
        get_prompt = lambda x: x["text"]
    return dataset, get_prompt

def load_chat_api_model(args):
    if args.model == "ChatGPT":
        client = OpenAI(api_key=args.api_key)
        api_call = lambda x: client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": x}],
            max_tokens=args.max_tokens,
        ).choices[0].message.content
    elif args.model == "Claude":
        client = Anthropic(api_key=args.api_key)
        api_call = lambda x: client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": x}],
            max_tokens=args.max_tokens,
        ).content[0].text
    elif args.model == "Grok":
        client = OpenAI(
            api_key=args.api_key,
            base_url="https://api.x.ai/v1",
        )
        api_call = lambda x: client.chat.completions.create(
            model="grok-beta",
            messages=[{"role": "user", "content": x}],
            max_tokens=args.max_tokens,
        ).choices[0].message.content
    elif args.model == "Gemini":
        genai.configure(api_key=args.api_key)
        model = genai.GenerativeModel("gemini-1.5-pro-002")
        api_call = lambda x: model.generate_content(x, generation_config={"max_output_tokens": args.max_tokens}).text
    elif args.model == "DeepSeek":
        client = OpenAI(
            api_key=args.api_key,            
            base_url="https://api.deepseek.com"
        )
        api_call = lambda x: client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": x}],
            max_tokens=args.max_tokens,
        ).choices[0].message.content
    return api_call

def generate_responses_chat_api(args):
    # form api call function
    api_call = load_chat_api_model(args)

    # load dataset
    dataset, get_prompt = create_dataset(args)

    # generate responses
    data = []
    random_indices = random.sample(range(len(dataset)), args.num_samples)
    for i in tqdm(random_indices, "Generating responses"):
        prompt = get_prompt(dataset[i])
        response = api_call(prompt)
        data.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])

        # save data per generation
        with open(args.output_path, "w") as file:
            json.dump(data, file)

def generate_responses_instruct_llm(args):
    # load model
    model, tokenizer = load_vllm_model(args)

    # load dataset
    dataset, get_prompt = create_dataset(args)

    random_indices = random.sample(range(len(dataset)), args.num_samples)
    all_prompts = []
    data = []
    for i in random_indices:
        prompt = get_prompt(dataset[i])
        data.append([{"role": "user", "content": prompt}])
        
        formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        all_prompts.append(formatted_prompt)
        
    outputs = model.generate(
        all_prompts,
        SamplingParams(
            temperature=args.temperature,
            max_tokens=1024,
        )
    )
    
    for i in range(args.num_samples):
        response = outputs[i].outputs[0].text
        # remove some artifacts
        if "mistral" in args.model:
            response = response.strip()
        elif "gemma" in args.model:
            response = response.rstrip('\n')
        data[i].append({"role": "assistant", "content": response})

    with open(args.output_path, "w") as file:
        json.dump(data, file) 

def generate_responses_base_llm(args):
    model, tokenizer = load_vllm_model(args)
    
    dataset, get_prompt = create_dataset(args)
    
    random_indices = random.sample(range(len(dataset)), args.num_samples)
    data = []
    all_prompts = []
    for i in random_indices:
        prompt = get_prompt(dataset[i])
        tokens = tokenizer.encode(prompt, padding=False)[:32]
        truncated_prompt = tokenizer.decode(tokens, skip_special_tokens=True)
        
        all_prompts.append(truncated_prompt)
        data.append([{"role": "user", "content": truncated_prompt}])

    outputs = model.generate(
        all_prompts,
        SamplingParams(
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_tokens=1024,
        )
    )
    
    for i in range(args.num_samples):
        data[i].append({"role": "assistant", "content": outputs[i].outputs[0].text})

    with open(args.output_path, "w") as file:
        json.dump(data, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # miscelaneous
    parser.add_argument("--seed", type=int, default=42, help="the seed that controls the randomness")
    parser.add_argument("--device", type=str, default="cuda", help="the device to use for generation")
    parser.add_argument("--num_gpus", type=int, default=2, help="the number of gpus to use for generation")
    parser.add_argument("--api_key", type=str, default=None, help="the api key to use for generation")
    
    # data
    parser.add_argument("--dataset", type=str, default="UltraChat", 
                        choices=["UltraChat", "Cosmopedia", "LmsysChat", "WildChat", "FineWeb"], 
                        help="the dataset to generate responses from")
    parser.add_argument("--model", type=str, default=None, 
                        choices=["ChatGPT", "Claude", "Grok", "Gemini", "DeepSeek", 
                                "Llama3-8b-it", "Llama3-70B-it", "Llama3-70B-it-FP8", "Llama3-70B-it-INT8",
                                "Gemma2-9b-it", "Gemma2-9b-it-FP8", "Gemma2-9b-it-INT8",
                                "Qwen2.5-7b-it", "Qwen2-72B-it", "Qwen2-72B-it-FP8", "Qwen2-72B-it-INT8",
                                "Mistral-7b-v3-it", "Mistral-7b-v3-it-FP8", "Mistral-7b-v3-it-INT8",
                                "Phi-4", "Llama3-8b-it-awq", "Llama3-8b-it-gptq",
                                "Llama3-8b", "Gemma2-9b", "Qwen2.5-7b", "Mistral-7b-v3"],
                        help="the model to generate responses from")
    parser.add_argument("--num_samples", type=int, default=11_000, help="the number of samples to generate")
    parser.add_argument("--output_path", type=str, default=None, help="the path to save the output")
    
    # sampling hyperparameters
    parser.add_argument("--temperature", type=float, default=0, help="the temperature of the sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="the repetition penalty of the sampling")
    parser.add_argument("--max_tokens", type=int, default=1024, help="the maximum number of tokens to generate")
    
    args = parser.parse_args()
    print(args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    if args.model in ["ChatGPT", "Claude", "Grok", "Gemini", "DeepSeek"]:
        generate_responses_chat_api(args)
    elif args.model in ["Llama3-8b-it", "Llama3-70B-it", "Llama3-70B-it-FP8", "Llama3-70B-it-INT8",
                       "Gemma2-9b-it", "Gemma2-9b-it-FP8", "Gemma2-9b-it-INT8",
                       "Qwen2.5-7b-it", "Qwen2-72B-it", "Qwen2-72B-it-FP8", "Qwen2-72B-it-INT8", 
                       "Mistral-7b-v3-it", "Mistral-7b-v3-it-FP8", "Mistral-7b-v3-it-INT8",
                       "Phi-4", "Llama3-8b-it-awq", "Llama3-8b-it-gptq"]:
        generate_responses_instruct_llm(args)
    else:
        assert args.dataset == "FineWeb"
        generate_responses_base_llm(args)