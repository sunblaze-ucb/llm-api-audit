import os
import json
import numpy as np
from tqdm import tqdm
import glob
import argparse

def evaluate_mmlu_sample(model_dir, n_samples=100, random_seed=42):
    np.random.seed(random_seed)
    
    # Load all examples
    all_examples = []
    subject_files = glob.glob(os.path.join(model_dir, "samples_mmlu_*.jsonl"))
    for file_path in subject_files:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    all_examples.append(json.loads(line))
        except Exception:
            continue
    
    if not all_examples:
        print("No MMLU examples found in directory")
        return None
        
    # Run Monte Carlo resampling
    accuracies = []
    for _ in tqdm(range(n_samples), desc="Monte Carlo runs"):
        correct_count = 0
        for example in all_examples:
            target = int(example['target'])
            filtered_resps = example['filtered_resps']
            all_log_probs = [float(resp[0]) for resp in filtered_resps]
            probs = np.exp(all_log_probs)
            probs = probs / np.sum(probs)
            sampled_choice = np.random.choice(len(probs), p=probs)
            if sampled_choice == target:
                correct_count += 1
        accuracies.append(correct_count / len(all_examples))
    
    # Calculate and print results
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)
    print(f"MMLU Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return mean_acc, std_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MMLU samples with Monte Carlo")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing MMLU sample files")
    parser.add_argument("--samples", type=int, default=100, help="Number of Monte Carlo samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    evaluate_mmlu_sample(args.dir, args.samples, args.seed)