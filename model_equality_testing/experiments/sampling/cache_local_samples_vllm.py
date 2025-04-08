"""
Caches n samples / prompt by locally inferencing a model on a dataset of prompts.
"""

from vllm import LLM, SamplingParams
import torch
import tqdm
import argparse
import os
import experiments.prompts as prompts_module
from experiments.utils import (
    str_to_bool,
    ParseKwargs,
    build_cache_filename,
)
import pickle
from accelerate import Accelerator
from experiments.sampling.model import TransformersModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_kwargs", nargs="*", action=ParseKwargs, default={})
    parser.add_argument(
        "--source",
        type=str,
        default="fp32",
    )
    parser.add_argument("--prompts", default="dummy", type=str)
    parser.add_argument("--do_sample", type=str_to_bool, default=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--L", type=int, default=50)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=90)
    parser.add_argument("--save_dir", type=str, default="../cache/samples")
    args = parser.parse_args()
    accelerator = Accelerator()
    print(args)

    # get dataset
    model_to_load_prompts = TransformersModel(
        args.model,
        accelerator=accelerator,
        skip_loading_weights=True,
    )
    ds = getattr(prompts_module, f"get_{args.prompts}_prompts")(model_to_load_prompts)
    try:
        # causes issues if kept
        ds = ds.remove_columns(["chat_tokens", "chat_with_ellipses_tokens"])
    except:
        pass

    # initialize vllm model
    # if args.source == "None":
    #     kwargs = {"dtype": "float32"}
    # elif args.source == "fp16":
    #     kwargs = {"dtype": "float16"}
    # elif args.source == "nf4":
    #     kwargs = {"quantization": "bitsandbytes", "load_format": "bitsandbytes"}
    # else:
    #     raise ValueError("Cannot use that with vllm")

    model = LLM(
        model=args.model,
        tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
        **args.model_kwargs,
        gpu_memory_utilization=0.95
    )

    # get save string
    filename = build_cache_filename(
        model=args.model,
        prompts=args.prompts,
        alternative=args.source,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        L=args.L,
    )

    sampling_params = SamplingParams(
        n=args.batch_size,
        temperature=args.temperature if args.temperature is not None else 1,
        top_p=args.top_p if args.top_p is not None else 1,
        top_k=1 if not args.do_sample else -1,
        max_tokens=args.L,
        stop_token_ids=[],
        skip_special_tokens=False,
        ignore_eos=True,
        logprobs=None,
    )

    # run through batches and dump the results
    for i in range(len(ds)):
        if os.path.exists(f"{args.save_dir}/{filename}-{i}.pkl"):
            print("Skipping", i)
            continue
        print(f"Collecting samples for prompt {i}")

        out = []
        for batch_size in tqdm.tqdm(
            [
                min(args.batch_size, args.n - j * args.batch_size)
                for j in range(
                    (args.n + args.batch_size - 1) // args.batch_size
                )
            ]
        ):
            out.extend(
                model.generate(
                    [ds["chat"][i]],
                    sampling_params,
                )
            )
        sample = [oi.token_ids for o in out for oi in o.outputs]

        with open(f"{args.save_dir}/{filename}-{i}.pkl", "wb") as f:
            pickle.dump(sample, f)