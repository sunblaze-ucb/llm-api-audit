from argparse import Action
from typing import List, Tuple, Union
import os
import torch
import torch.nn.functional as F
from ast import literal_eval
import time
import numpy as np
import hashlib

FILE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def build_cache_filename(
    *,
    filename: str = None,
    model: str = None,
    prompts: Union[str, List[str]] = None,
    alternative: Union[str, Tuple[str, str]] = None,
    use_char_space: bool = None,
    temperature: float = None,
    top_p: float = None,
    do_sample: bool = True,
    L: int = None,
    stat: str = None,
    N: Union[int, Tuple[int]] = None,
    prompt_indices_json: str = None,
):
    """
    Build the filename used for logits, parametric bootstrap, power caches, etc.
    Appends to the given filename if provided.
    Format:
    {model}-{prompts}-{alternative}-{temperature}-{top_p}-{do_sample}-{L}-{stat}-{N}-{prompt_indices}
    """
    if filename is None:
        filename = ""
    if model is not None:
        filename += f"{sanitize(model)}"
    if prompts is not None:
        charspace = "-char" if use_char_space == True else ""
        if isinstance(prompts, list):
            prompts = "-".join(sorted(prompts))
            if (
                prompts
                == "wikipedia_de-wikipedia_en-wikipedia_es-wikipedia_fr-wikipedia_ru"
            ):
                prompts = "wikipedia"
        filename += f"-{prompts}{charspace}"
    if alternative is not None:
        if type(alternative) == tuple:
            assert (
                len(alternative) == 2
            ), "Only support a tuple in the case of a composite null with two simple nulls"
            filename += f"-{sanitize(alternative[0])}_{sanitize(alternative[1])}"  # note: None will be shown in the composite case
        else:
            filename += (
                f"-{sanitize(alternative)}" if alternative != "None" else ""
            )  # None will not be shown in the simple case
    if temperature is not None:
        filename += f"-temp={temperature}"
    if top_p is not None:
        filename += f"-top_p={top_p}"
    if do_sample is False:
        filename += "-greedy"
    if L is not None:
        filename += f"-L={L}"
    if stat is not None:
        filename += f"-{stat}"
    if N is not None:
        if type(N) == tuple:
            assert len(N) == 2, "N should only be a tuple for two-sample testing"
            N = f"{N[0]}_{N[1]}"
        filename += f"-n={N}"
    if prompt_indices_json is not None:
        # convert prompt_indices_json to relative filepath from root
        to_hash = os.path.abspath(prompt_indices_json).replace(FILE_DIR, "")
        setting = hash_fn(to_hash)
        filename += f"-prompt_indices={setting}"
    return filename


###################################
# Argparse utils
###################################


def str_to_bool(value):
    """
    Function to parse boolean values from argparse.
    """
    if value.lower() in ("true", "t", "yes", "y"):
        return True
    elif value.lower() in ("false", "f", "no", "n"):
        return False
    else:
        raise Exception(f"Invalid boolean value: {value}")


class ParseKwargs(Action):
    """
    Helper function s.t. argparse can parse kwargs of the form --kwarg key1=value1 key2=value2
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for pair in values:
            key, value = pair.split("=")
            processed_value = infer_type(value)
            getattr(namespace, self.dest)[key] = processed_value


def infer_type(s):
    """
    If the str can be interpreted as a float or an int, convert it to that type.
    """
    try:
        return str_to_bool(s)
    except:
        pass
    try:
        return literal_eval(s)
    except:
        pass
    try:
        return str_to_torchdtype(s)
    except:
        pass
    try:
        return str_to_list(s)
    except:
        return s


def str_to_torchdtype(value):
    if not value.startswith("torch."):
        raise Exception(f"Invalid torch dtype: {value}")
    return getattr(torch, value.split(".")[1])


def str_to_list(value):
    """
    Helper function to parse a string of the form "[x,y,z]" into a list [x, y, z].
    Catches some cases where ast.literal_eval fails because the elements in the list
    contain non-standard characters.
    """
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    else:
        raise Exception(f"Invalid list value: {value}")

    return value.split(",")


def str_to_bool(value):
    """
    Function to parse boolean values from argparse.
    """
    if value.lower() in ("true", "t", "yes", "y"):
        return True
    elif value.lower() in ("false", "f", "no", "n"):
        return False
    else:
        raise Exception(f"Invalid boolean value: {value}")


###################################
# Misc utils
###################################


def collate(list_of_dicts):
    """
    Collate a list of dictionaries into a single dictionary.
    """
    collated_dict = defaultdict(list)
    for d in list_of_dicts:
        for k, v in d.items():
            collated_dict[k].append(v)
    return collated_dict


def uncollate(dict_of_lists, step=1):
    """
    Uncollate a dictionary of lists into a list of dictionaries.
    If step > 1, the output will be a list of dicts where each dict key
    has values that are themselves lists of size step.
    If step = 1, the outputs will be a list of dicts where each dict value
    are not lists.
    """
    list_of_dicts = []
    num_elements = len(dict_of_lists[list(dict_of_lists.keys())[0]])

    for i in range(0, num_elements, step):
        dict_entry = {}
        for key, value_list in dict_of_lists.items():
            _n = min(i + step, num_elements) - i
            if value_list is None:
                dict_entry[key] = ([None] * _n) if _n > 1 else None
            else:
                dict_entry[key] = value_list[i : i + step] if _n > 1 else value_list[i]
        list_of_dicts.append(dict_entry)

    return list_of_dicts


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


def wait_if_error(
    callable,
    *args,
    timeout=1,
    max_retries=5,
    exception_if_fail=False,
    special_exceptions=[],
    **kwargs,
):
    """
    Call a function, and if it raises an exception, wait for timeout seconds and try again,
    up to max_retries times.
    """
    for try_number in range(max_retries):
        try:
            return callable(*args, **kwargs)
        except Exception as e:
            if type(e) in special_exceptions:
                print(f"Error: {e}; finishing on a special exception")
                return None
            print(f"Error: {e}; waiting {timeout} seconds and trying again")
            time.sleep(timeout ** (try_number + 1))

    if exception_if_fail:
        raise Exception(f"Failed after {max_retries} tries")
    else:
        print(f"Failed after {max_retries} tries")
        return None


def seed_everything(seed: int):
    """
    Helper function to seed everything.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StreamingDataset(torch.utils.data.Dataset):
    """
    Truncates a stream dataset at num_samples.
    """

    def __init__(self, stream_dataset, num_samples, char_limit=None):
        self.stream_dataset = stream_dataset
        self.num_samples = num_samples
        self.samples = list(self._truncated_dataset(char_limit=char_limit))

    def _truncated_dataset(self, char_limit=None):
        i = 0
        seen = set()
        for sample in self.stream_dataset:
            if i == self.num_samples:
                break
            if sample["id"] in seen:
                continue
            if char_limit is None or len(sample["plain"]) <= char_limit:
                i += 1
                seen.add(sample["id"])
                yield sample

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # if integer, get the row
        # if string, get the column
        if isinstance(idx, (int, slice)):
            return self.samples[idx]
        elif isinstance(idx, str):
            return [sample[idx] for sample in self.samples]
        else:
            raise ValueError("idx must be an integer or a string")

    def remove_columns(self, columns):
        for sample in self.samples:
            for col in columns:
                sample.pop(col, None)
        return self


def hash_fn(x: object, type="md5"):
    """
    Hash an object determinisitically.
    """
    # encode the object
    if isinstance(x, torch.Tensor):
        encoded = x.numpy().tobytes()
    elif isinstance(x, np.ndarray):
        encoded = x.tobytes()
    elif isinstance(x, str):
        encoded = x.encode("utf-8")
    else:
        encoded = pickle.dumps(x)
    # hash the encoded object
    if type == "md5":
        return hashlib.md5(encoded).hexdigest()
    elif type == "sha256":
        return hashlib.sha256(encoded).hexdigest()
    else:
        return zlib.adler32(encoded)
