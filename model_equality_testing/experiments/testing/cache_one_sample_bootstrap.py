"""
Calls model_equality_testing.pvalue.one_sample_parametric_bootstrap_pvalue repeatedly to cache simulated test statistics (parametric bootstrap) for a given null x test statistic.
"""

from model_equality_testing.dataset import load_distribution
from model_equality_testing.pvalue import one_sample_parametric_bootstrap_pvalue
import pickle
from experiments.utils import build_cache_filename, str_to_bool
import argparse
import os
from typing import List
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--null_distribution_source",
        type=str,
        default="fp32",
        help="Source to use as the null distribution",
    )
    parser.add_argument(
        "--prompt_indices_json",
        type=str,
        help="JSON of prompt dataset name: [list of indices]. Prompt distribution will be uniform over these prompts",
        required=True,
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Number of samples to draw each time we compute the test statistic",
        default=None,
    )
    parser.add_argument(
        "--n_per_prompt",
        type=int,
        help="Alternative to --n, number of samples per prompt",
        default=None,
    )
    parser.add_argument(
        "--stat", type=str, default="g_squared", help="One-sample test statistic"
    )
    parser.add_argument(
        "--test_in_unicode",
        type=str_to_bool,
        default=True,
        help="Test in unicode space instead of token space",
    )
    parser.add_argument(
        "--b", type=int, default=1000, help="Number of bootstrap samples"
    )
    parser.add_argument("--L", type=int, default=1000, help="Maximum completion length")
    parser.add_argument("--do_sample", type=str_to_bool, default=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument(
        "--save_dir", type=str, default="../cache/parametric_bootstrap_stats"
    )
    args = parser.parse_args()
    print(args)

    # Load the prompt indices
    js = json.load(open(args.prompt_indices_json))
    prompts = list(js.keys())
    # get the number of prompts m
    m = sum([len(js[k]) for k in js])
    print("Number of prompts: ", m)

    assert (args.n is not None) ^ (
        args.n_per_prompt is not None
    ), "Exactly one of --n or --n_per_prompt must be provided"
    if args.n_per_prompt is not None:
        args.n = m * args.n_per_prompt
    print("Number of samples: ", args.n)

    # Load dataset
    p = load_distribution(
        model=args.model,
        prompt_ids=js,
        L=args.L,
        source=args.null_distribution_source,
        load_in_unicode=args.test_in_unicode,
    )
    print("Null shape: ", p.shape)

    # Construct the filename
    filename = build_cache_filename(
        model=args.model,
        prompts=prompts,
        prompt_indices_json=args.prompt_indices_json,
        alternative=args.null_distribution_source,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        L=args.L,
        stat=args.stat,
        N=args.n,
        use_char_space=args.test_in_unicode,
    )
    out_path = f"{args.save_dir}/{filename}.pkl"
    print(f"Will write to {out_path}")

    # Skip if already exists
    if os.path.exists(out_path):
        print("Skipping...")
        exit()

    # Cache the test statistics
    _, s = one_sample_parametric_bootstrap_pvalue(
        null_dist=p,
        n=args.n,
        b=args.b,
        return_stats=True,
        stat_type=args.stat,
    )

    with open(out_path, "wb") as f:
        pickle.dump(s, f)
