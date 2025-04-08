import numpy as np
import torch
from typing import Union, Tuple, List, Dict
from model_equality_testing.utils import (
    ndim,
    stack_with_padding,
    Stopwatch,
)
import os
from functools import lru_cache, cache

FILE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sample_from_categorical(
    p: Union[np.ndarray, torch.Tensor, List[torch.Tensor]],
    n: Union[int, List[int], np.ndarray, torch.Tensor] = 1,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Samples n indices according to the distribution given by p
    Args:
        p: (m, k) tensor or m-list of (ki,) tensors giving the probabilities of each
            of the k items for each of the m prompts
        n: number of samples to draw per prompt (along the first dimension)
            can be a scalar or a list of ints
    Returns:
        sample:
            if n is a scalar, an (m, n) tensor
            otherwise, a m-list of tensors, where each tensor is (ni,)
    """
    if isinstance(n, (list, torch.Tensor, np.ndarray)):
        assert len(n) == len(p), "n must be a scalar or have the same length as p"
    if isinstance(p, torch.Tensor):
        # strategy: draw the max n samples so that we can parallelize, and then truncate
        # since p is a tensor, directly use torch.multinomial
        max_n = max(n) if isinstance(n, (list, torch.Tensor, np.ndarray)) else n
        sample = torch.multinomial(p, max_n, replacement=True)
        sample = sample.unsqueeze(-1)
        if isinstance(n, (list, torch.Tensor, np.ndarray)) and not all(
            ni == max_n for ni in n
        ):
            sample = [si[:ni] for si, ni in zip(sample, n)]
    else:
        # since p is a (potentially jagged) list, we need to iterate over each prompt
        if isinstance(n, (list, torch.Tensor, np.ndarray)):
            sample = [
                torch.multinomial(pi, ni, replacement=True).unsqueeze(-1)
                for pi, ni in zip(p, n)
            ]
        else:
            sample = [
                torch.multinomial(pi, n, replacement=True).unsqueeze(-1) for pi in p
            ]

    if isinstance(sample, list):
        try:
            return torch.stack(sample)
        except:
            return sample
    else:
        return sample


########################
# Distribution objects #
########################


class CompletionSample:
    def __init__(self, prompts: Union[np.ndarray, torch.Tensor], completions: Union[np.ndarray, torch.Tensor], m: int):
        """
        Represents a sample from a DistributionFromDataset object.
        Args:
            prompts: a (N,) tensor or array of prompt indices
            completions: a (N, L) tensor or array of completions, where L is the maximum completion length
                i.e. this is pre-padded. prompts[i] should correspond to the prompt for completions[i]
            m: total number of prompts; used to enforce that the prompt indices are in [0, m)
        """
        self.m = m

        if isinstance(prompts, np.ndarray):
            prompts = torch.from_numpy(prompts)
        if isinstance(completions, np.ndarray):
            completions = torch.from_numpy(completions)

        self.prompt_sample = prompts.clone().detach()
        self.completion_sample = completions.clone().detach()
        if self.completion_sample.ndim == 1:
            self.completion_sample = self.completion_sample.unsqueeze(-1)

        self.L = self.completion_sample.shape[-1]
        self.N = len(self.prompt_sample)
        self.ns = torch.tensor(
            [(self.prompt_sample == i).sum().item() for i in range(m)]
        )  # effective number of completions for each prompt
        assert self.completion_sample.ndim == 2
        assert self.prompt_sample.ndim == 1

    @property
    def shape(self):
        return (self.m, self.ns.tolist(), self.L)

    @property
    @cache
    def sample(self):
        """(m, n, L) tensor or len-m list of (ni, L) tensors"""
        return self._prompt_completion_to_sample(
            self.prompt_sample, self.completion_sample
        )

    @property
    @cache
    def sequences(self):
        return torch.cat(
            [self.prompt_sample.unsqueeze(1), self.completion_sample], dim=1
        )

    def __str__(self):
        return f"CompletionSample with n={self.ns}"

    def __repr__(self):
        return str(self.sequences)

    @cache
    def _prompt_completion_to_sample(self, prompt_sample, completion_sample):
        """
        Converts view 1 to view 2
        """
        assert len(prompt_sample) == len(completion_sample) == self.N
        max_n = max(self.ns)
        if all([ni == max_n for ni in self.ns]):
            sample = torch.zeros(self.m, max_n, self.shape[-1], dtype=int)
        else:
            sample = [torch.zeros(ni, self.shape[-1], dtype=int) for ni in self.ns]
        count_up = torch.zeros(self.m, dtype=int)
        for i, (prompt, completion) in enumerate(zip(prompt_sample, completion_sample)):
            sample[prompt][count_up[prompt].item()] = completion
            count_up[prompt] += 1
        return sample

    def _sample_to_prompt_completion(self, sample):
        """
        Converts view 2 to view 1
        """
        assert ndim(sample) == 3
        assert len(sample) == self.m
        if isinstance(sample, torch.Tensor):
            indices = []
            for i, tensor in enumerate(sample):
                indices.extend([i] * len(tensor))
            indices = torch.tensor(indices)
            completion_sample = sample.view(-1, sample.shape[-1])
        else:
            indices = []
            for i, tensor in enumerate(sample):
                indices.extend([i] * len(tensor))
            indices = torch.tensor(indices)
            completion_sample = torch.cat(sample)
        return indices, completion_sample


class DistributionFromDataset:
    """
    Helper class to create a prompt-completion distribution from a dataset of text samples.
    Given a dataset of completions for each prompt, we can create a distribution over
    prompt-completion pairs by sampling with replacement from the completions for each prompt.
    """

    def __init__(
        self,
        sample_paths: List[Tuple[str, callable]],
        L: int,
        prompt_distribution: Union[np.ndarray, torch.Tensor] = None,
        logprob_paths: List[Tuple[str, callable]] = None,
        pad_token_id: int = -1,
    ):
        """
        Args:
            sample_paths: list of tuples of paths to completion files and a callable to load them
                The length of the list is the number of prompts m.
                Example: sample_paths = [
                    ("completions-to-prompt-1-file-500-samples", load_fn),
                    ("completions-to-prompt-2-file-200-samples", load_fn),
                    ("completions-to-prompt-2-file-500-samples", load_fn),
                ]
                => DistributionFromDataset(m=3, k=[500, 200, 500], L=self.L)
                The callable should return an integer numpy array of shape (N, L') and pad with
                the same pad_token_id passed into this class if any completions are shorter than L'
                Example:
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, self.pad_token_id]])
            L: maximum completion length
                If the length of completions in files is L' > L, we truncate to L
            prompt_distribution: (m,) tensor
                The distribution over prompts
                If None, defaults to uniform
            logprob_paths: list of tuples of paths to logprob files and a callable to load them
                The length of the list is the number of prompts m.
                All completions that we load from the sample_paths should be in the logprob files
                The callable should return a dictionary of completions (tuples) to numpy arrays of
                logprobs, where the length of the key is the same length as the value.
                Example:
                {
                    (0, 1, 2): [logprob_0, logprob_1, logprob_2],
                }
            pad_token_id: the id of the padding token
                If the length of completions in files is L' < L, we pad with this token
        """
        self.sample_paths = sample_paths
        self.logprob_paths = logprob_paths
        self.m = len(sample_paths)
        self.L = L
        if prompt_distribution is None:
            prompt_distribution = torch.ones(self.m) / self.m
        self.prompt_distribution = prompt_distribution
        self.pad_token_id = pad_token_id

        # first call to _load_samples: establish ki, the number of possible completions for each prompt
        print("Initializing DistributionFromDataset; loading k...")
        self.k = torch.tensor(
            [len(self._load_samples(i, print_info=True)) for i in range(self.m)]
        )
        print("Done initializing")

    def __len__(self):
        return self.m

    @property
    def shape(self):
        return (self.m, self.k.tolist(), self.L)

    @lru_cache(maxsize=100)
    def _load_samples(self, i, print_info=False) -> torch.Tensor:
        """
        Loads the cached samples for the given prompt i based on the sample_paths passed
        in at initialization
        Recall that sample_paths is of type List[Tuple[str, callable]]
        The callable should return an integer numpy array of shape (k, L') and pad with
        the same pad_token_id passed into this class if any completions are shorter than L'
        Example:
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, self.pad_token_id]])
        Args:
            i: the index of the prompt
        Returns:
            samples: (k, L) tensor
        """
        path, load_fn = self.sample_paths[i]
        with Stopwatch() as sw:
            try:
                sample = load_fn(path)  # should be a numpy array
                sample = torch.from_numpy(sample)
            except Exception as e:
                raise ValueError(f"Error loading {path} for prompt {i}: {e}")
        if print_info:
            print(
                f"\tTime to load distribution {i} w/ {len(sample)} entries: {sw.time}"
            )
        if len(sample) == 0:
            print(f"Warning: sample {i} is empty")
        else:
            # truncate to L
            sample = stack_with_padding(
                sample[:, : self.L],
                padding_token=self.pad_token_id,
            )[0]
            # pad to L
            sample = torch.cat(
                [
                    sample,
                    self.pad_token_id
                    * torch.ones((len(sample), (self.L - sample.shape[1])), dtype=int),
                ],
                dim=-1,
            )
        return sample

    @lru_cache(maxsize=100)
    def _load_probs(self, i, print_info=False) -> dict:
        """
        Loads the cached probs for the given prompt i based on the logprob_paths passed
        in at initialization
        Recall that logprob_paths is of type List[Tuple[str, callable]]
        The callable should return a dictionary of completions (tuples) to numpy arrays of
        logprobs, where the length of the key is the same length as the value.
        Example:
        {
            (0, 1, 2): [logprob_0, logprob_1, logprob_2],
        }
        Args:
            i: the index of the prompt
        Returns:
            d: dictionary of completions (tuples) to probs (floats)
        """
        path, load_fn = self.logprob_paths[i]
        with Stopwatch() as sw:
            d = load_fn(
                path
            )  # map from completions (as tuples of ints) to probs (floats)
        # truncate to L
        new_d = {}
        for k, v in d.items():
            key = tuple(
                k[:self.L] + (self.pad_token_id, ) * (self.L - min(self.L, len(k)))
            )
            new_d[key] = v[: self.L].sum().exp().item()
        d = new_d

        if print_info:
            print(f"\tTime to load distribution {i} w/ {len(d)} entries: {sw.time}")
            print(f"Sum of cached probs is {np.sum(list(d.values()))}")
        return d

    def sample(self, n: int = 1, prompt_indices=None) -> CompletionSample:
        """
        Samples n prompt-completion pairs according to the joint distribution
        P(x, y) = P(x) * P(y | x)
        by using the following two-step process:
        1. Sample a prompt x from the prompt distribution
        2. Sample a completion y from the completion distribution given x
        The completion distribution is assumed to be uniform over the completions
        loaded from the dataset; we sample with replacement.
        Args:
            n: number of samples to draw overall
        """
        if prompt_indices is not None:
            assert 0 <= min(prompt_indices) and max(prompt_indices) < self.m
        else:
            prompt_indices = list(range(self.m))

        # First, sample from the prompt distribution
        prompt_samples = _sample_from_categorical(
            self.prompt_distribution, n=n
        ).squeeze()

        # Then, count how many completions we need for each prompt
        prompt_indices, prompt_counts = torch.unique(prompt_samples, return_counts=True)
        prompt_indices = prompt_indices.tolist()
        prompt_counts = prompt_counts.tolist()

        # Finally, sample from the completion distribution for each prompt
        samples = []  # note: will be ordered by prompt index
        for i, ni in zip(prompt_indices, prompt_counts):
            s = self._load_samples(i)  # (ki, L) tensor
            samples.append(s[np.random.choice(len(s), ni, replace=True)])
        assert len(samples) == len(prompt_indices)

        # Create a CompletionSample object
        sample = CompletionSample(
            prompts=torch.cat(
                [torch.tensor([i] * ni) for i, ni in zip(prompt_indices, prompt_counts)]
            ),  # prompt samples (N,)
            completions=torch.cat(samples),  # completion samples (N,)
            m=self.m,
        )
        return sample

    def get_completion_probabilities(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Given a list of n sequences [(x, y_1, ...., y_L)]
        return an (n,) array of [P(y_1, ..., y_L | x)]
        by looking up the completion in the output of self._load_probs(x)
        Args:
            sequences: (n, L+1) tensor of sequences
                where the first column is the prompt index
        Returns:
            probabilities: (n,) tensor of probabilities
        """
        assert sequences.ndim == 2
        final_probs = torch.zeros(len(sequences), dtype=self.prompt_distribution.dtype)
        prompts = sequences[:, 0]

        for prompt_index in range(self.m):
            mask = prompts == prompt_index
            if mask.sum() == 0:
                continue
            try:
                d = self._load_probs(prompt_index)
            except:
                raise ValueError(
                    "Cannot get completion probabilities because could not find logprob files."
                )
            vals = torch.tensor([d[tuple(row[1:].tolist())] for row in sequences[mask]])
            final_probs[mask] = vals
        return final_probs

    def get_all_completion_probabilities(self, i):
        """
        Enumerate all completions (y_1, ..., y_L) given x=i and their probabilities
        P(y_1, ..., y_L | x=i)
        Args:
            i (int): the prompt index
        Returns:
            completions: (n, L) tensor
            p: (n,) tensor
        """
        try:
            d = self._load_probs(i)
        except:
            raise ValueError(
                "Cannot get completion probabilities because could not find logprob files."
            )
        sequences_i = torch.tensor(list(d.keys()))
        p_i = torch.tensor(list(d.values()))
        return sequences_i, p_i

    @cache
    def get_all_joint_probabilities(self):
        """
        Enumerate all joint sequences (x, y_1, ..., y_L) and their joint probabilities
        P(x, y_1, ..., y_L) = P(x) * P(y_1, ..., y_L | x)
        Returns:
            unique: (ntilde, L+1) tensor of joint sequences
            probs: (ntilde,) tensor of joint probabilities
        """
        sequences, p = [], []
        for i in range(self.m):
            sequences_i, p_i = self.get_all_completion_probabilities(i)
            sequences.append(sequences_i)
            p.append(p_i)
        probs = torch.cat(
            [p[i] * self.prompt_distribution[i] for i in range(self.m)]
        ).view(-1)
        unique = torch.cat(
            [
                torch.tensor([[i] + list(s) for s in sequences_i])
                for i, sequences_i in enumerate(sequences)
            ]
        )
        return unique, probs

    def __del__(self):
        self._load_samples.cache_clear()
        self._load_probs.cache_clear()
        self.get_all_joint_probabilities.cache_clear()
