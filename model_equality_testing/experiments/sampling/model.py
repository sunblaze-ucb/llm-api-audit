"""
Helper functions for working with Huggingface Transformers models.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    WatermarkingConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    TopKLogitsWarper,
    WatermarkLogitsProcessor,
)
from contextlib import suppress
import json
from experiments.utils import collate, uncollate, stack_with_padding
from typing import List, Literal, Union
import os
from dataclasses import dataclass, asdict
import warnings
from accelerate import load_checkpoint_and_dispatch, Accelerator
from accelerate.state import AcceleratorState
import numpy as np
import tqdm
from torch.utils.data import DataLoader

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

Role = Literal["system", "user", "assistant", "instruction"]


@dataclass
class Message:
    role: Role
    content: str


Dialog = List[Message]


class TransformersModel:
    """
    Wraps a Huggingface Transformers model with helper methods.
    """

    def __init__(
        self,
        model_path: str,
        accelerator: Accelerator,
        checkpoint: str = None,
        tokenizer_path: str = None,
        device: Union[int, torch.device] = -1,
        cast_dtype=torch.float32,
        quantize: Literal[4, 8, None] = None,
        autocast_dtype=None,
        accelerate_model_parallelism: bool = False,  # not deepspeed compatible
        generation_config_path: str = f"{FILE_DIR}/../constants/vanilla_local.json",
        fixed_decoding_params: dict = {},
        skip_loading_weights: bool = False,
        watermark_bias: float = None,
        batch_size: int = 80,
    ):
        """
        Args:
            - model_path: the path to the Huggingface model
            - checkpoint: the path to a checkpoint to load
            - tokenizer_path: the path to the tokenizer
            - device: the device to use for inputs
            - cast_dtype: the dtype to cast the model to
            - quantize: whether to quantize the model
            - autocast_dtype: the dtype to use for autocasting
            - accelerate_model_parallelism: whether to use accelerate's native model parallelism
        """
        self.model_name = model_path
        self.batch_size = batch_size

        # special kind of checkpoint: pretrained ckpt
        if is_pretrained_ckpt(checkpoint):
            model_path = checkpoint
            checkpoint = None

        # load model
        quantization_config = (
            BitsAndBytesConfig(
                load_in_8bit=(quantize == 8),
                load_in_4bit=(quantize == 4),
                bnb_4bit_quant_type="nf4",
                llm_int8_has_fp16_weight=False,
            )
            if quantize is not None
            else None
        )
        token = os.environ["HF_TOKEN"] if "HF_TOKEN" in os.environ else None
        if not skip_loading_weights:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=cast_dtype,
                quantization_config=quantization_config,
                device_map=("auto" if accelerate_model_parallelism else None),
                token=token,
                trust_remote_code=True,
            )
            print("loaded")
            self.model.eval()
        else:
            self.model = None
        self._is_quantized = quantize != None

        # prepare with accelerator
        assert accelerator is not None  # for this script
        self._accelerator = accelerator
        self.model = self._accelerator.prepare(self.model)
        print(self._accelerator.distributed_type)
        self._model_handle = self.model.module if self._accelerator.distributed_type == "MULTI_GPU" else self.model

        # print for debugging
        if not skip_loading_weights:
            print(self.model)
        if torch.cuda.is_available():
            print(
                "GPU memory still available:", torch.cuda.mem_get_info()[0] // 1024**2
            )

        # autocast
        self.autocast = (
            suppress()
            if autocast_dtype is None
            else torch.cuda.amp.autocast(dtype=autocast_dtype)
        )

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name if tokenizer_path is None else tokenizer_path,
            trust_remote_code=True,
        )
        if "Llama-3.1" in (self.model_name if tokenizer_path is None else tokenizer_path):
            with open(f"{FILE_DIR}/../constants/llama3-chat-template.txt") as f:
                self.tokenizer.chat_template = f.read()
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        ## pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if hasattr(self._model_handle, "config") and isinstance(self._model_handle.config, dict):
            # this can sometimes happen if the model being loaded is a pretrained ckpt
            self._model_handle.config["pad_token_id"] = self.tokenizer.pad_token_id
        elif hasattr(self._model_handle, "config"):
            self._model_handle.config.pad_token_id = self.tokenizer.pad_token_id
        ## vocab size
        if hasattr(self._model_handle, "config") and hasattr(self._model_handle.config, "vocab_size"):
            self.vocab_size = self._model_handle.config.vocab_size
            self.tokenizer._vocab_size = self.vocab_size
        else:
            self.vocab_size = len(self.tokenizer)
            self.tokenizer._vocab_size = self.vocab_size

        if generation_config_path is not None:
            self._gen_cfg = json.load(open(generation_config_path, "r"))
        else:
            warnings.warn(
                "Models sometimes have default generation configs that are not what we expect. Check the model's documentation."
            )
            self._gen_cfg = {}
        self._fixed_gen_cfg = fixed_decoding_params
        if watermark_bias is not None:
            print("Adding watermarking with bias", watermark_bias)
            self._watermark_cfg = WatermarkingConfig(
                bias=watermark_bias, seeding_scheme="lefthash", context_width=1
            )
        else:
            self._watermark_cfg = None

    def __str__(self):
        return self.model_name

    def eval(self):
        self.model.eval()

    def state_dict(self):
        return self._model_handle.state_dict()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.__call__(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    ########################

    def format_as_chat(
        self,
        prompt: str,
        system_message: str = None,
        tokenize: bool = False,
        add_ellipses: bool = False,
        ellipses: str = '"...',
        remove_special: bool = True,
    ):
        """
        Formats a plain prompt into a chat prompt, and adds ellipses as the start of the continuation
        Args:
            - prompt: the prompt to format
            - system_message: the system message to include
            - remove_special: whether to remove special tokens if not tokenizing
                If True, the prompt will be returned without the bos token that most chat templates add
                The goal is that later calls to the tokenizer with add_special_tokens=True will produce the same
                output as if this function was called with remove_special_tokens=False.
        """
        # Special case: no chat template
        if self.tokenizer.chat_template is None:
            assert (
                system_message is None
            ), "System message not supported for a model without chat templates."
            if add_ellipses:
                prompt = prompt + "\n" + ellipses
            if tokenize:
                return self.tokenizer.encode(prompt, add_special_tokens=True)
            elif not remove_special:
                return self.tokenizer.decode(
                    self.tokenizer.encode(prompt, add_special_tokens=True),
                    skip_special_tokens=False,
                )
            else:
                return prompt

        # Normal case: with chat template
        d = (
            [Message(role="system", content=system_message)]
            if system_message is not None
            else []
        )
        d += [Message(role="user", content=prompt)]
        d = [asdict(di) for di in d]

        if not tokenize:
            prompt = self.tokenizer.apply_chat_template(
                d, tokenize=False, add_generation_prompt=True
            )
            if remove_special and self.tokenizer.bos_token is not None:
                prompt = prompt.replace(self.tokenizer.bos_token, "", 1)
            if add_ellipses:
                if self.tokenizer.encode(prompt + " " + ellipses) == (
                    self.tokenizer.encode(prompt)
                    + self.tokenizer.encode(ellipses, add_special_tokens=False)
                ):
                    prompt = prompt + " " + ellipses
                else:
                    prompt = prompt + ellipses
            return prompt

        # tokenize
        prompt = self.tokenizer.apply_chat_template(
            d, tokenize=True, add_generation_prompt=True
        )
        if add_ellipses:
            ellipses = self.tokenizer.encode(ellipses, add_special_tokens=False)
            prompt = prompt + ellipses
        return prompt

    ######## INFERENCE METHODS #########

    def generate(
        self,
        prompts: List[str],
        return_decoded: bool = True,
        return_collated: bool = False,
        truncation: int = None,
        add_special_tokens: bool = True,
        skip_special_tokens: bool = False,
        return_scores: bool = False,
        **decode_kwargs,
    ):
        """
        Calls generate() on the model and returns with the prompts removed, optionally returning logits.
        Args:
            - prompts (List[str]): the prompts to generate from
                shape (B,)
            - return_decoded: whether to return the decoded text under the 'text' key
            - return_collated: whether to return the output as a collated dictionary
                vs. a list of dictionaries
            - truncation: the maximum length of the output
            - add_special_tokens: whether to add special tokens
            - skip_special_tokens: whether to skip special tokens in the output
            - decode_kwargs: additional arguments to pass to generate()
        Returns a B-list of dictionaries with the following keys:
            - 'sequences': the output token ids
            - 'text': the decoded text if return_decoded is True
        """
        all_decode_kwargs = {**self._gen_cfg, **decode_kwargs, **self._fixed_gen_cfg}
        for k in self._fixed_gen_cfg:
            assert (
                all_decode_kwargs[k] == self._fixed_gen_cfg[k]
            ), f"Should have been fixed to {self._fixed_gen_cfg[k]} but was {all_decode_kwargs[k]}"
        print("Decoding with", all_decode_kwargs)

        # generate
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=(truncation is not None),
            max_length=truncation,
            add_special_tokens=add_special_tokens,
        ).to(self._accelerator.device)

        with torch.inference_mode():
            with self.autocast:
                raw = self._model_handle.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"].bool(),
                    **all_decode_kwargs,
                    watermarking_config=self._watermark_cfg,
                    return_dict_in_generate=True,
                    output_scores=return_scores,
                    logits_processor=self._get_logits_warpers(
                        return_custom_only=True,
                    ),
                )

        # slice to just the output
        output = {
            "sequences": raw.sequences[:, inputs["input_ids"].shape[1] :],
            "scores": (
                torch.log_softmax(torch.stack(raw.scores, dim=1), dim=-1)
                if return_scores
                else None
            ),
        }
        if return_decoded:
            output["text"] = self.tokenizer.batch_decode(
                output["sequences"], skip_special_tokens=skip_special_tokens
            )
        num_out_per_prompt = all_decode_kwargs.get("num_return_sequences", 1)
        output = uncollate(
            output, step=num_out_per_prompt
        )  # outer dimension is over prompt

        if return_collated:
            return collate(output)
        return output

    def sample(
        self,
        prompts: List[str],
        n: Union[int, List[int]] = 1,
        L: int = 1,
        return_logprobs=False,
        **kwargs,
    ):
        """
        Samples n sequences per prompt of length L each.
        Args:
            - prompts: the prompts to sample from
                Shape: (B,)
            - n: the number of samples to generate per prompt
                Shape can be (B,) or a single integer
            - L: the length of each sample
            - max_batch_size: the maximum batch size to use
            - kwargs: additional arguments to pass to generate()
        Returns the completion token ids.
            Shape is (B, N, L)
        """
        max_batch_size = self.batch_size

        # we want to take advantage of parallelism, so we sample the maximum n for each prompt
        assert not isinstance(n, (list, torch.Tensor, np.ndarray)) or len(n) == len(
            prompts
        )
        max_n = max(n) if isinstance(n, (list, torch.Tensor, np.ndarray)) else n
        if torch.is_tensor(max_n):
            max_n = max_n.item()

        # break into batches and sample
        # if accelerator is set, parallelize
        num_processes = self._accelerator.num_processes
        max_n = int(np.ceil(max_n / num_processes))
        sample, scores = [], [] # outer dim is over batch
        for batch_size in tqdm.tqdm(
            [
                min(max_batch_size, max_n - i * max_batch_size)
                for i in range((max_n + max_batch_size - 1) // max_batch_size)
            ]
        ):
            out = self.generate(
                prompts,
                max_new_tokens=L,
                num_return_sequences=batch_size,
                do_sample=True,
                return_scores=return_logprobs,
                **kwargs,
            )
            # make sure out contains two dimensional sequences: N x L
            sequences = [
                torch.cat(
                    [
                        (
                            o["sequences"]
                            if o["sequences"].ndim == 2
                            else o["sequences"].unsqueeze(0)
                        ),
                        self.tokenizer.eos_token_id
                        * torch.ones((batch_size, L - o["sequences"].shape[-1])).to(o['sequences'].device),
                    ],
                    dim=1,
                )
                .long().cpu()
                for o in out
            ] # outer dim over batch, inner over (n, L)
            sample.append(sequences)  # stack across prompts
            if return_logprobs:
                batch_scores = [
                    torch.gather(
                        o["scores"],
                        -1,
                        o["sequences"].unsqueeze(-1),
                    ).squeeze(-1).cpu()
                    for o in out
                ]
                batch_scores = [x.unsqueeze(0) if x.ndim == 1 else x for x in batch_scores]
                scores.append(batch_scores)  # stack across prompts

        sample, mask = zip(*[
            stack_with_padding(
                [s[i] for s in sample], 
                padding_token=self.tokenizer.pad_token_id
            )
            for i in range(len(prompts)) # across prompts
        ])
        sample = [s.view(-1, s.shape[-1]) for s in sample]
        mask = [m.view(-1, m.shape[-1]) for m in mask]
        sample = torch.stack(sample, dim=0) # (B, N, L)
        mask = torch.stack(mask, dim=0)
        if return_logprobs:
            scores, _ = zip(*[
                stack_with_padding(
                    [s[i] for s in scores], 
                    padding_token=self.tokenizer.pad_token_id
                )
                for i in range(len(prompts)) # across prompts
            ])
            scores = torch.cat(scores, dim=0)

        # cut out the extra samples
        if isinstance(n, (list, torch.Tensor, np.ndarray)):
            sample = [s[:ni] for ni, s in zip(n, sample)]
            if return_logprobs:
                scores = [s[:ni] for ni, s in zip(n, scores)]
            try:
                sample = torch.stack(sample, dim=0)
                scores = torch.stack(scores, dim=0)
            except:
                pass
        else:
            sample = sample[:, :n]
            if return_logprobs:
                scores = scores[:, :n]
        if return_logprobs:
            return sample, scores
        return sample

    def get_logits(
        self,
        prompts: List[str] = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        add_special_tokens: bool = True,
        truncation: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
    ):
        """
        Given prompts (x_0, x_1, ..., x_{ell-1}), either as input_ids or as a string,
        returns the logits for x_{ell}
        Shape of input_ids is (B, ell-1)
        Shape of output is (B, k)
        """
        max_batch_size = self.batch_size
        logits_warpers = self._get_logits_warpers(temperature, top_k, top_p)

        # tokenize prompts
        assert (prompts is not None) ^ (input_ids is not None)
        if prompts is not None:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                truncation=(truncation is not None),
                max_length=truncation,
                add_special_tokens=add_special_tokens,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

        loader = DataLoader(
            list(zip(np.arange(len(input_ids)), input_ids, attention_mask)),
            batch_size=max_batch_size,
            shuffle=False,
        )
        loader = self._accelerator.prepare_data_loader(loader)

        # get logits
        all_logits = []
        ids = []
        for batch_ids, batch_input_ids, batch_attention_mask in tqdm.tqdm(
            loader, disable=not self._accelerator.is_main_process
        ):
            with torch.no_grad():
                logits = self.forward(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                ).logits
            logits = logits[:, -1, :].detach()
            logits = logits_warpers(input_ids, logits)
            all_logits.append(logits)
            ids.append(batch_ids)

        all_logits = torch.cat(all_logits, dim=0) # (B, k)
        ids = torch.cat(ids, dim=0)
        all_logits = self._accelerator.gather(all_logits).cpu()
        ids = self._accelerator.gather(ids).cpu()

        # sort by original order, there may be duplicates
        _, idx = np.unique(ids.numpy(), return_index=True)
        all_logits = all_logits[idx]
        return all_logits
    
    def get_logprobs(
        self,
        prompts: List[str] = None,
        prompt_input_ids: torch.Tensor = None,
        prompt_attention_mask: torch.Tensor = None,
        completions: List[str] = None,
        completion_input_ids: torch.Tensor = None,
        completion_attention_mask: torch.Tensor = None,
        add_special_tokens: bool = True,
        truncation: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
    ):
        """
        Gets the log probabilities of completions given prompts.
        """
        assert (prompts is not None) ^ (prompt_input_ids is not None and prompt_attention_mask is not None)
        assert (completions is not None) ^ (completion_input_ids is not None and completion_attention_mask is not None)
        max_batch_size = self.batch_size

        # concatenate prompts and completions
        if prompt_input_ids is None:
            prompt_inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                truncation=(truncation is not None),
                max_length=truncation,
                add_special_tokens=add_special_tokens,
            )
            prompt_input_ids = prompt_inputs["input_ids"]
            prompt_attention_mask = prompt_inputs["attention_mask"]
        if completion_input_ids is None:
            completion_inputs = self.tokenizer(
                completions,
                return_tensors="pt",
                padding="longest",
                truncation=(truncation is not None),
                max_length=truncation,
                add_special_tokens=False,
            )
            completion_input_ids = completion_inputs["input_ids"]
            completion_attention_mask = completion_inputs["attention_mask"]
        
        input_ids = torch.cat([prompt_input_ids, completion_input_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)

        loader = DataLoader(
            list(zip(np.arange(len(input_ids)), input_ids, attention_mask, completion_attention_mask)),
            batch_size=max_batch_size,
            shuffle=False,
        )
        loader = self._accelerator.prepare_data_loader(loader)
        
        # loop through and get logprobs
        logits_warpers = self._get_logits_warpers(temperature, top_k, top_p)

        all_logprobs = []
        ids = []
        for batch_ids, batch_input_ids, batch_attention_mask, batch_completion_mask in tqdm.tqdm(
            loader, disable=not self._accelerator.is_main_process
        ):

            # use forward pass to get logits
            with torch.no_grad():
                logits = self.forward(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                ).logits

            # pass through warpers; try to match what would happen in generate()
            new_logits = []
            L = completion_input_ids.shape[1]
            for i in range(L):
                i += prompt_input_ids.shape[1] - 1
                new_logits.append(logits_warpers(
                    batch_input_ids[:, :i+1],
                    logits[:, i, :],
                ))
            logits = torch.stack(new_logits, dim=1)

            # get logprobs of the completion tokens
            # for pad tokens in completion, set logprob to 0 so that sum() works
            logprobs = torch.log_softmax(logits, dim=-1)
            gen_probs = torch.gather(
                logprobs, 2, batch_input_ids[:, -L:].unsqueeze(-1).to(logprobs.device)
            ).squeeze(-1) # (B, completion len padded)
            all_logprobs.extend([
                logprob[mask[-L:]].cpu()
                for logprob, mask in zip(gen_probs, batch_completion_mask.bool())
            ])
            ids.extend(batch_ids.cpu())

        all_logprobs = np.array(all_logprobs)
        _, idx = np.unique(np.array(ids), return_index=True)
        all_logprobs = all_logprobs[idx]
        return all_logprobs
    
    def _get_logits_warpers(self, temperature=None, top_k=None, top_p=None, return_custom_only=False):
        logits_warpers = LogitsProcessorList()
        if temperature is not None and not return_custom_only:
            logits_warpers.append(TemperatureLogitsWarper(temperature=temperature))
        if top_k is not None and not return_custom_only:
            logits_warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
        if top_p is not None and not return_custom_only:
            logits_warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
        if self._watermark_cfg is not None and not return_custom_only:
            logits_warpers.append(
                WatermarkLogitsProcessor(
                    vocab_size=self.vocab_size,
                    device=self._accelerator.device,
                    **dict(self._watermark_cfg),
                )
            )
        return logits_warpers

#################################################

def is_pretrained_ckpt(path):
    """
    Given a path to a checkpoint, determine if it is a pretrained checkpoint,
    i.e. saved with model.save_pretrained().
    """
    if path is None:
        return False
    files = os.listdir(path) if os.path.isdir(path) else [path]
    if any("generation_config.json" in f for f in files):
        return True
    return False
