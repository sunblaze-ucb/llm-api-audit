"""
Caches n samples / prompt from an API on a dataset of prompts.
"""

import tqdm
import os
import argparse
import experiments.prompts as prompts_module
from experiments.sampling.model import TransformersModel
from experiments.utils import (
    str_to_bool,
    build_cache_filename,
    wait_if_error,
)
import experiments.sampling.api as api_module
import time
from dataclasses import dataclass, asdict
import time
from accelerate import Accelerator
import json
import pickle

"""
Logic to choose how to query APIs; this was selected to try to account for different tokenization policies by APIs. Set by manually testing whether the number of tokens in the prompt is the same as the number of prompt tokens mentioned in the returned message.
"""


# case-by-case policies to handle uniform tokenization. use test_api_tokenization.py to set these
def policy(api, model):
    if api in ["replicate"]:
        default = {
            "prompt_key": "chat",
            "use_chat_endpoint": False,
            "expected_prompt_key": "chat",
        }
    elif api == "amazon":
        default = {
            "prompt_key": "chat_with_special",
            "use_chat_endpoint": False,
            "expected_prompt_key": "chat",
        }
    else:
        default = {
            "prompt_key": "plain",
            "use_chat_endpoint": True,
            "expected_prompt_key": "chat",
        }
    return default


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(f"{FILE_DIR}/../constants/minimum_prompt_indices.json") as f:
    BARE_MINIMUM = json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="HF transformers model name"
    )
    parser.add_argument("--backend", type=str, required=True, help="API name")
    parser.add_argument("--prompts", default="dummy")
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument(
        "--n", type=int, default=1, help="Number of samples to generate per prompt"
    )
    parser.add_argument("--do_sample", type=str_to_bool, default=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--val_cutoff", type=int, default=20)
    parser.add_argument(
        "--sample_bare_minimum",
        type=str_to_bool,
        default=False,
        help="Whether to only sample the bare minimum prompt indices used for the prompt distributions in the paper, listed in constants/minimum_prompt_indices.json",
    )
    parser.add_argument("--save_dir", type=str, default="../cache/api")
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

    setup_kwargs = policy(args.backend, args.model)

    # backend
    try:
        get_fn, kwargs = getattr(api_module, f"setup_{args.backend}")(
            model=args.model,
            N=1,
            L=args.L,
            use_chat_endpoint=setup_kwargs["use_chat_endpoint"],
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    except AttributeError:
        raise ValueError("Unrecognized backend")

    filename = build_cache_filename(
        model=args.model,
        prompts=args.prompts,
        alternative=args.backend,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        L=args.L,
    )
    filename = f"{args.save_dir}/{filename}"

    # get samples
    for it, x in tqdm.tqdm(enumerate(ds)):
        if (
            args.sample_bare_minimum
            and args.prompts in BARE_MINIMUM
            and it not in BARE_MINIMUM[args.prompts]
        ):
            continue
        if os.path.exists(f"{filename}-{it}.pkl"):
            print("Skipping, exists...")
            continue

        print("Sampling", it)
        time.sleep(30) # avoid rate limiting

        out = []
        for i in range(args.n):
            print("> sample #", i)
            o = wait_if_error(
                get_fn, timeout=10, prompt=x[setup_kwargs["prompt_key"]], **kwargs
            )
            if o is not None:
                out.extend(o)
        out = [asdict(o) for o in out]
        d = {
            "samples": out,
            "id": x["id"],
            "prompt": x[setup_kwargs["prompt_key"]],
            "y": x.get("y", None),
        }

        with open(f"{filename}-{it}.pkl", "wb") as f:
            pickle.dump(d, f)
