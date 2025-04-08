from typing import List, Tuple, Union
import torch
import torch.nn.functional as F
from time import perf_counter
import numpy as np
from functools import lru_cache

def tokenize_unicode(strings: List[str], pad_token_id: int = -1) -> np.ndarray:
    """
    Tokenize a list of strings into a list of lists of unicode codepoints,
    and then stack them into a 2D numpy array, using -1 as padding.

    Args:
        strings (List[str]): list of strings to tokenize

    Returns:
        List[List[int]]: list of lists of unicode codepoints
    """
    strings = [torch.tensor([ord(c) for c in s]) for s in strings]
    chr_array = stack_with_padding(strings, padding_token=-1)[0].numpy()
    return chr_array

def pad_to_length(samples: np.ndarray, L: int, pad_token_id: int = -1) -> np.ndarray:
    """
    Pad a 2D numpy array of samples to a fixed length L using a pad token.

    Args:
        samples (np.ndarray): 2D numpy array of samples
        L (int): length to pad to
        pad_token_id (int): padding token

    If the current length is longer than L, throws an error.

    Returns:
        np.ndarray: padded 2D numpy array
    """
    if samples.shape[1] > L:
        raise ValueError(f"Cannot pad to length {L} because the current length is {samples.shape[1]}")
    padded_samples = np.full((samples.shape[0], L), pad_token_id)
    padded_samples[:, :samples.shape[1]] = samples
    return padded_samples

def sanitize(s):
    """Sanitize a string for use as a filename."""
    s = str(s)
    s = s.replace(" ", "-")
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace(",", "_")
    s = s.replace("/", "-")
    s = s.replace("(", "")
    s = s.replace(")", "")
    return s


def stack_with_padding(
    tensors: List[torch.Tensor],
    dim: int = 0,
    padding_side: str = "right",
    padding_mode: str = "constant",
    padding_token: Union[int, float] = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stack tensors along specified dimension and pad them to ensure their size is equal in all dimensions.
    Returns the stacked tensor and a boolean mask indicating valid (non-padded) elements.

    Args:
        tensors (List[torch.Tensor]): list of tensors to stack
        dim (int): dimension along which to stack tensors. Defaults to 0.
        padding_side (str): side on which to pad - "left" or "right". Defaults to "right".
        padding_mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        padding_value (Union[int, float]): value to use for constant padding

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: stacked tensor and boolean mask
    """
    # Ensure all tensors have the same number of dimensions
    max_dims = max(t.dim() for t in tensors)
    tensors = [t.view(*t.shape, *([1] * (max_dims - t.dim()))) for t in tensors]

    # Find the maximum size for each dimension
    max_sizes = [max(t.shape[i] for t in tensors) for i in range(max_dims)]

    def make_padding(tensor_shape):
        padding = []
        for i in reversed(range(max_dims)):  # Reverse for F.pad expectations
            pad_size = max_sizes[i] - tensor_shape[i]
            if padding_side == "left":
                padding.extend([pad_size, 0])
            elif padding_side == "right":
                padding.extend([0, pad_size])
            else:
                raise ValueError(f"padding_side '{padding_side}' is unknown")
        return tuple(padding)

    padded_tensors = []
    masks = []

    for t in tensors:
        padding = make_padding(t.shape)
        padded_t = F.pad(t, padding, mode=padding_mode, value=padding_token)

        mask = torch.zeros_like(padded_t, dtype=torch.bool)
        slices = []
        for i in range(max_dims):
            if padding_side == "left":
                slices.append(slice(max_sizes[i] - t.shape[i], None))
            else:
                slices.append(slice(None, t.shape[i]))
        mask[tuple(slices)] = True

        padded_tensors.append(padded_t)
        masks.append(mask)

    stacked_tensor = torch.stack(padded_tensors, dim=dim)
    stacked_mask = torch.stack(masks, dim=dim)

    return stacked_tensor, stacked_mask


class Stopwatch:
    """
    Context manager for timing a block of code
    Source: https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
    """

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.time = perf_counter() - self.time


def ndim(p):
    """
    Args:
        p: either a tensor or a list of tensors or a list of lists of tensors
    """
    if isinstance(p, torch.Tensor):
        return p.ndim
    if not isinstance(p, (list, np.ndarray)):
        return 0
    elif len(p) > 0:
        return ndim(p[0]) + 1
    else:
        return 1


@lru_cache(maxsize=100)
def get_inv(lst):
    """
    Given a list of items, returns a dict of {item: ix}
    """
    return {x: idx for idx, x in enumerate(lst)}
