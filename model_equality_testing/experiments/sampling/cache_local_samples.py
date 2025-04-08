"""
Caches n samples / prompt by locally inferencing a model on a dataset of prompts.
"""

import torch
import tqdm
import argparse
import os
import experiments.prompts as prompts_module
from experiments.sampling.model import TransformersModel
from experiments.utils import (
    str_to_bool,
    ParseKwargs,
    build_cache_filename,
)
from accelerate import Accelerator
import pickle
import glob


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
    
    # setup model
    fixed_decoding_params = {
        "temperature": args.temperature if args.temperature is not None else 1,
        "top_p": args.top_p if args.top_p is not None else 1,
    }
    kwargs = {}
    if args.source == "fp16":
        kwargs = {"cast_dtype": torch.float16}
    elif args.source == "int8":
        kwargs = {"quantize": 8}
    elif args.source == "nf4":
        kwargs = {"quantize": 4}
    elif args.source == "watermark":
        kwargs = {"watermark_bias": 2.5}
    model = TransformersModel(
        args.model,
        accelerator=accelerator,
        fixed_decoding_params=fixed_decoding_params,
        batch_size=args.batch_size,
        **args.model_kwargs,
        **kwargs
    ) 

    # load dataset
    ds = getattr(prompts_module, f"get_{args.prompts}_prompts")(model)
    try:
        # causes issues if kept
        ds = ds.remove_columns(["chat_tokens", "chat_with_ellipses_tokens"])
    except:
        pass

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

    # run through batches and dump the results
    for i in range(len(ds["chat"])):
        if os.path.exists(f"{args.save_dir}/{filename}-{i}.pkl"):
            print("Skipping", i)
            continue
        print(f"Collecting samples for prompt {i}")

        out = model.sample(
            [ds["chat"][i]],
            n=args.n,
            L=args.L,
        ) # (1, n, L)
        out = out.squeeze().tolist() # (n, L)
        with open(f"{args.save_dir}/{filename}-{i}.pkl", "wb") as f:
            pickle.dump(out, f)
