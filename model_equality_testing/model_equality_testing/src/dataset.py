"""
Helper functions for downloading and loading the dataset we release.
"""

from model_equality_testing.distribution import DistributionFromDataset
from typing import Literal, List, Dict, Union, Tuple
from model_equality_testing.utils import sanitize, stack_with_padding, tokenize_unicode
import glob
import numpy as np
import torch
from transformers import AutoTokenizer
import pickle
from collections import defaultdict
import os
import zipfile

def download_dataset(root_dir="./data"):
    """
    Download the dataset from Google Drive and extract it into the specified directory.

    Arguments:
    - root_dir (str): The root directory where the dataset will be saved.
    - split (str or None): 'test', 'val', or None. If None, both splits will be downloaded.
    """
    try:
        import gdown
    except:
        raise ImportError("Please install gdown to download the dataset: pip install gdown")
    
    os.makedirs(root_dir, exist_ok=True)
    file_id = "1csgp83tx04kVA9ejp6MNSC7Ni0vh0J9U"
    download_url = f"https://drive.google.com/uc?id={file_id}"

    zip_file_path = os.path.join(root_dir, "model_equality_testing_ds.zip")
    extract_dir = root_dir

    print(f"Downloading dataset...")
    if not os.path.exists(zip_file_path):
        gdown.download(download_url, zip_file_path, quiet=False)

    print(f"Extracting dataset...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(zip_file_path)
    print(f"Dataset downloaded and extracted successfully.")


SOURCES = [
    "fp32",
    "fp16",
    "nf4",
    "int8",
    "watermark",
    "anyscale",
    "amazon",
    "fireworks",
    "replicate",
    "deepinfra",
    "groq",
    "perplexity",
    "together",
    "azure",
]

MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def load_distribution(
    model: str,
    prompt_ids: dict,
    L: int,
    source: str,
    prompt_distribution: np.ndarray = None,
    load_in_unicode: bool = True,
    root_dir="./data",
) -> DistributionFromDataset:
    """
    Given a dictionary mapping {dataset_name: [prompt_ids]}, load the dataset and return a DistributionFromDataset object.
    Args:
        model: a string representing the model name
        prompt_ids: a dictionary mapping {dataset_name: [prompt_ids]}
            where dataset_name is a string and prompt_ids is a list of strings.
            Example:
            {
                "wikipedia_en": [0, 1, 4],
                "wikipedia_es": [5],
            }
            => DistributionFromDataset(m=4, L=L)
        L: completion length
        source: a string representing
    Returns:
        a DistributionFromDataset object, which allows for sampling from the dataset.
    """
    assert source in SOURCES, f"source must be one of {SOURCES}"
    print(f"Loading dataset from source: {source}")
    print(f"Prompt IDs: {prompt_ids}")

    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if source in ["fp32", "fp16", "nf4", "int8"]:
        if load_in_unicode:
            load_fn = lambda x: _load_local_samples_unicode(x, tokenizer)
        else:
            load_fn = lambda x: _load_local_samples_tokens(x, tokenizer)
    else:
        assert load_in_unicode, "Only unicode supported for API samples"
        load_fn = lambda x: _load_api_samples_unicode(x, tokenizer)

    filenames_and_callables = []
    logprob_filenames_and_callables = []
    for ds, ids in prompt_ids.items():
        for id in ids:
            name = f"{sanitize(model)}-{ds}-{source}-L=*-{id}"
            lst = glob.glob(f"{root_dir}/samples/{name}.pkl")
            if len(lst) == 0:
                print(
                    f"Warning: no pkl file matching the pattern {name} was found in the specified dataset directory {root_dir}.",
                    "Make sure the `root_dir` variable is set correctly and that `model` and `prompt_ids` exist in the dataset."
                )
            fn = lst[0]
            filenames_and_callables.append((fn, load_fn))

            # logprobs: include if not in unicode and the file exists
            if not load_in_unicode:
                try:
                    fn = glob.glob(f"{root_dir}/logprobs/{name}.pkl")[0]
                    logprob_filenames_and_callables.append((fn, lambda x: _load_logprobs(x, tokenizer)))
                except IndexError:
                    pass

    return DistributionFromDataset(
        sample_paths=filenames_and_callables,
        L=L,
        prompt_distribution=prompt_distribution,
        logprob_paths=logprob_filenames_and_callables,
        pad_token_id=tokenizer.pad_token_id if not load_in_unicode else -1,
    )


def _pad_after_first_eos(array, eos_token_id, pad_token_id):
    """
    In each sequence, converts all tokens to the right of the first occurrence
    of the eos_token to a pad_token.
    Edge case: does not pad if the eos token is the first token in the sequence
    """
    assert array.ndim == 2
    eos_locs = np.expand_dims((array == eos_token_id).astype(float).argmax(axis=1), 1)
    eos_locs[eos_locs == 0] = array.shape[1]
    col_indices = np.repeat(
        np.expand_dims(np.arange(array.shape[1]), 0), array.shape[0], axis=0
    )
    array[col_indices > eos_locs] = pad_token_id
    return array


def _load_local_samples_tokens(path, tok) -> np.ndarray:
    """
    Loads local samples into a (k, max_L) array of token ids.
    Assumes samples are saved as lists of lists of token ids (.pkl files).
    """
    with open(path, "rb") as f:
        array = pickle.load(f)
    array = np.array(array)
    array = _pad_after_first_eos(array, tok.eos_token_id, tok.pad_token_id)
    return array


def _load_local_samples_unicode(path, tok) -> np.ndarray:
    """
    Loads local samples into a (k, max_L) array of character ids (using Python's ord function).
    Assumes samples are saved as lists of token ids (.pkl files).
    Uses -1 as a padding token in the output.
    """
    with open(path, "rb") as f:
        array = pickle.load(f)
    array = np.array(array)
    array = _pad_after_first_eos(array, tok.eos_token_id, tok.pad_token_id)
    strings = tok.batch_decode(array, skip_special_tokens=True)
    return tokenize_unicode(strings)


def _load_api_samples_unicode(path, tok) -> np.ndarray:
    """
    Loads API samples from disk where each character is assigned an id, rather than in token space.
    """
    with open(path, "rb") as f:
        js = pickle.load(f)
    samples = js["samples"]
    samples = [torch.tensor([ord(c) for c in s["full_completion"]]) for s in samples]
    chr_array = stack_with_padding(samples, padding_token=-1)[0].numpy()
    return chr_array


def _load_logprobs(path, tok) -> dict:
    """
    Returns a dictionary of {prompt: {sequence: logprob}} as saved
    by cache_logprobs.py or cache_distribution_by_sampling.py using utils.dump.
    """
    with open(path, "rb") as f:
        logprobs = pickle.load(f)

    # merge dictionaries with the same keys
    new_dict = {}
    for k, v in logprobs.items():
        k = list(k)
        v = torch.tensor(v)
        try:
            ix = k.index(tok.eos_token_id)
            k[ix + 1 :] = [tok.pad_token_id] * (len(k) - ix - 1)
            v[ix + 1 :] = 0
        except:
            pass
        new_dict[tuple(k)] = v
    return new_dict
