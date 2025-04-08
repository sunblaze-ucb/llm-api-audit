import os
import json
import numpy as np
import torch
from datetime import datetime
import tqdm
import argparse
from experiments.testing.simulation import get_power_two_sample
from model_equality_testing.dataset import load_distribution
from model_equality_testing.distribution import DistributionFromDataset, CompletionSample

def mix_distribution(
    test_distribution: DistributionFromDataset,
    target_distribution: DistributionFromDataset,
    p: float,
) -> DistributionFromDataset:
    assert 0 <= p <= 1, "p must be between 0 and 1"
    assert test_distribution.L == target_distribution.L, "Distributions must have the same completion length L"
    
    class MixedDistribution(DistributionFromDataset):
        def __init__(self, test_dist, target_dist, p):
            self.sample_paths = test_dist.sample_paths
            self.logprob_paths = test_dist.logprob_paths
            self.m = test_dist.m
            self.L = test_dist.L
            self.prompt_distribution = test_dist.prompt_distribution
            self.pad_token_id = test_dist.pad_token_id
            self.k = test_dist.k
            self.test_dist = test_dist
            self.target_dist = target_dist
            self.p = p
        
        def sample(self, n=1, **kwargs):
            import numpy as np
            choices = np.random.random(n) < self.p
            n_target = np.sum(choices)
            n_test = n - n_target
            
            if n_test == n:
                return self.test_dist.sample(n=n, **kwargs)
            
            elif n_target == n:
                return self.target_dist.sample(n=n, **kwargs)
            
            else:
                test_samples = self.test_dist.sample(n=int(n_test), **kwargs)
                target_samples = self.target_dist.sample(n=int(n_target), **kwargs)
                
                combined_prompts = torch.cat([test_samples.prompt_sample, target_samples.prompt_sample])
                combined_completions = torch.cat([test_samples.completion_sample, target_samples.completion_sample])
                
                shuffle_indices = torch.randperm(n)
                combined_prompts = combined_prompts[shuffle_indices]
                combined_completions = combined_completions[shuffle_indices]
                
                return CompletionSample(
                    prompts=combined_prompts,
                    completions=combined_completions,
                    m=self.m
                )
        
        def __getattr__(self, name):
            return getattr(self.test_dist, name)
    
    return MixedDistribution(test_distribution, target_distribution, p)

def run_sequential_experiment(model_name, p_value, n_simulations=100, n_null=250, n_data=250, b=1000):
    model_short_name = model_name.split('/')[-1]
    output_dir = f"{model_short_name}_p{p_value}"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_prompts = {
        "wikipedia_ru": [20, 21, 22, 23, 24, 25],
        "wikipedia_en": [20, 21, 22, 23, 24, 25, 26],
        "wikipedia_es": [20, 21, 22, 23, 24, 25],
        "wikipedia_fr": [20, 21, 22, 23, 24, 25]
    }
    
    fp32_distribution = load_distribution(
        model=model_name,
        prompt_ids=dataset_prompts,
        L=1000,
        source="fp32",
        load_in_unicode=True,
        root_dir="./data",
    )
    
    if p_value == 0.0:
        data_dist = fp32_distribution
    elif p_value == 1.0:
        data_dist = load_distribution(
            model=model_name,
            prompt_ids=dataset_prompts,
            L=1000,
            source="int8",
            load_in_unicode=True,
            root_dir="./data",
        )
    else:
        int8_distribution = load_distribution(
            model=model_name,
            prompt_ids=dataset_prompts,
            L=1000,
            source="int8",
            load_in_unicode=True,
            root_dir="./data",
        )
        data_dist = mix_distribution(
            test_distribution=fp32_distribution,
            target_distribution=int8_distribution,
            p=p_value
        )
    
    results = []
    for sim_idx in tqdm.tqdm(range(n_simulations)):
        try:
            power, rejections, pvalues, stats = get_power_two_sample(
                null_dist=fp32_distribution,
                data_dist=data_dist,
                n_null=n_null,
                n_data=n_data,
                n_simulations=1,
                alpha=0.05,
                pvalue_type="permutation_pvalue",
                stat_type="mmd_hamming",
                b=b,
                return_pvalue=True,
                return_stat=True,
            )
            
            result = {
                "model": model_name,
                "p": p_value,
                "rejection": bool(rejections[0])
            }
            
            results.append(result)
                
        except Exception:
            pass
    
    power = sum(1 for r in results if r.get("rejection", False)) / len(results) if results else 0
    
    summary = {
        "model": model_name,
        "p": p_value,
        "power": power
    }
    
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--p", type=float, required=True)
    
    parser.add_argument("--n_simulations", type=int, default=100)
    parser.add_argument("--n_null", type=int, default=250)
    parser.add_argument("--n_data", type=int, default=250)
    parser.add_argument("--b", type=int, default=1000)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    summary = run_sequential_experiment(
        model_name=args.model,
        p_value=args.p,
        n_simulations=args.n_simulations,
        n_null=args.n_null,
        n_data=args.n_data,
        b=args.b
    )