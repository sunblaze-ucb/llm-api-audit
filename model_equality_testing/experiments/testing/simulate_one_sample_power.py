"""
Simulate power of a one-sample test.
"""

from model_equality_testing.dataset import load_distribution
from experiments.testing.simulation import get_power_one_sample
from experiments.utils import build_cache_filename, str_to_bool
from experiments.testing.bootstrap_manager import BootstrapManager
import argparse
import os
from typing import List
import json
import argparse
import os
import torch
import numpy as np
from accelerate import Accelerator

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
        help="Source to use as the null distribution P",
    )
    parser.add_argument(
        "--alternative_distribution_source",
        type=str,
        default="fp32",
        help="Source to use as the alternative distribution Q",
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=100,
        help="Number of simulations to run to estimate power",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
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
        "--pvalue_type",
        type=str,
        default="parametric_bootstrap",
        help="Type of p-value",
    )
    parser.add_argument(
        "--test_in_unicode",
        type=str_to_bool,
        default=True,
        help="Test in unicode space instead of token space",
    )
    parser.add_argument("--L", type=int, default=1000, help="Maximum completion length")
    parser.add_argument("--max_b", type=int, default=None)
    parser.add_argument("--min_b", type=int, default=None)
    parser.add_argument("--do_sample", type=str_to_bool, default=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--save_dir", type=str, default="../cache/power")
    parser.add_argument("--bootstrap_dir", type=str, default="../cache/parametric_bootstrap_stats")
    accelerator = Accelerator()
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

    # Load null distribution
    p = load_distribution(
        model=args.model,
        prompt_ids=js,
        L=args.L,
        source=args.null_distribution_source,
        load_in_unicode=args.test_in_unicode,
    )
    print("Null shape: ", p.shape)

    # Start building the output filename
    filename_stem = build_cache_filename(
        model=args.model,
        prompts=prompts,
        prompt_indices_json=args.prompt_indices_json,
        use_char_space=args.test_in_unicode,
        alternative=args.null_distribution_source,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        L=args.L,
        stat=args.stat,
        N="{n}",
    )
    filename = filename_stem.format(n=args.n)
    filename += f"-pvalue={args.pvalue_type}-alt={args.alternative_distribution_source}"

    if os.path.exists(f"{args.save_dir}/{filename}.pt"):
        print("Skipping b/c already exists...")
        exit()

    # Set up bootstrap manager
    get_pvalue = None
    if args.pvalue_type == "parametric_bootstrap":
        try:
            bootstrap_manager = BootstrapManager(
                bootstrap_path_template=f"{args.bootstrap_dir}/{filename_stem}.pkl",
                min_b=args.min_b,
                max_b=args.max_b,
            )
            get_pvalue = bootstrap_manager.load(n=args.n)
        except:
            get_pvalue = None

    # Load alternative distribution
    q = load_distribution(
        model=args.model,
        prompt_ids=js,
        L=args.L,
        source=args.alternative_distribution_source,
        load_in_unicode=args.test_in_unicode,
    )
    print("Alternative shape: ", q.shape)

    # Simulate!
    (
        pwr,
        reject_history,
        alpha_history,
        pvalue_history,
        test_stats,
    ) = get_power_one_sample(
        p,
        q,
        n=args.n,
        n_simulations=args.n_simulations,
        pvalue_type=args.pvalue_type,
        stat_type=args.stat,
        get_pvalue_fn=get_pvalue,
        return_pvalue=True,
        return_alpha=True,
        return_stat=True,
        alpha=args.alpha,
    )

    # Save results
    print("Power: ", pwr)
    torch.save(
        {
            "power": pwr,
            "reject_history": reject_history,
            "alpha_history": alpha_history,
            "pvalue_history": pvalue_history,
            "test_stats": test_stats,
        },
        f"{args.save_dir}/{filename}.pt",
    )
