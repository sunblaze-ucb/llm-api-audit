"""
Given a set of samples, computes the logprobs of the individual completion tokens (one number per token) under the fp32 null.
Used in dataset construction; later applied for goodness-of-fit testing.
Assumes that samples are saved as *-{i}.pkl files, where i is the index of the prompt in the dataset,
and that the dataset is implemented in experiments.prompts.
"""

import torch
import tqdm
import glob
import pickle
import argparse
import os
import experiments.prompts as prompts_module
from experiments.sampling.model import TransformersModel
from experiments.utils import (
    str_to_bool,
    ParseKwargs,
    stack_with_padding,
)
from accelerate import Accelerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Model to get logprobs with"
    )   
    parser.add_argument("--model_kwargs", nargs="*", action=ParseKwargs, default={})
    parser.add_argument(
        "--prompts", type=str, required=True, help="Name of prompts dataset to get logprobs for"
    )
    parser.add_argument("--samples_path_template", type=str, required=True, help="String template for pkl file of samples, saved as list of lists of tokens ids. The first part of {template}-{i}.pkl")
    parser.add_argument("--do_sample", type=str_to_bool, default=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=90)
    parser.add_argument("--save_dir", type=str, default="../cache/logprobs")
    args = parser.parse_args()
    accelerator = Accelerator()
    print(args)

    # setup model
    fixed_decoding_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    model = TransformersModel(
        args.model,
        accelerator=accelerator,
        fixed_decoding_params=fixed_decoding_params,
        batch_size=args.batch_size,
        **args.model_kwargs
    ) 

    # load dataset
    ds = getattr(prompts_module, f"get_{args.prompts}_prompts")(model)
    try:
        # causes issues if kept
        ds = ds.remove_columns(["chat_tokens", "chat_with_ellipses_tokens"])
    except:
        pass

    # run through batches and dump the results
    for path in tqdm.tqdm(glob.glob(f"{args.samples_path_template}-*")):
        i = int(os.path.basename(path).split("-")[-1].split(".")[0])
        if os.path.exists(f"{args.save_dir}/{os.path.basename(args.samples_path_template)}-{i}.pkl"):
            print("Skipping", path)
            continue
        
        print(f"Collecting logprobs for prompt {i} based on path {path}")

        with open(path, "rb") as f:
            # note: this expects to pkl file to contain a list of lists of integers (token IDs)
            completions = [torch.tensor(x) for x in pickle.load(f)]

        completions, attention_mask = stack_with_padding(completions)

        logprobs = model.get_logprobs(
            prompts=[ds[i]["chat"]] * len(completions),
            completion_input_ids=completions,
            completion_attention_mask=attention_mask,
        )
        with open(f"{args.save_dir}/{os.path.basename(args.samples_path_template)}-{i}.pkl", "wb") as f:
            out = {tuple(seq.tolist()): lp.tolist() for seq, lp in zip(completions, logprobs)}
            pickle.dump(out, f)