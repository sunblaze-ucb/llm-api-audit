"""
Library of functions to setup API clients for sampling.
The exposed functions are of the form setup_{api}.
They return a tuple of three functions:
    - get_fn: the function to call the API
    - kwargs: the default decoding kwargs to pass to the API
"""

from transformers import AutoTokenizer
import os
import openai
import time
import numpy as np
from dataclasses import dataclass, asdict
import re
import time
import requests
from typing import List
import json

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
default_decoding_kwargs = json.load(open(f"{FILE_DIR}/../constants/vanilla_api.json"))


@dataclass
class APIOut:
    """
    Dataclass for the output of a single API sample
    """

    prompt: str = None
    prompt_token_ids: List[int] = None
    full_completion: str = None
    completion_token_ids: List[int] = None
    completion_tokens: List[str] = None
    completion_token_logprobs: List[float] = None
    full_logprob: float = None
    L: int = None
    num_prompt_tokens: int = None
    latency_ms: int = None


def setup_anyscale(
    model: str,
    N: int,
    L: int,
    use_chat_endpoint: bool = False,
    do_sample: bool = True,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
):
    client = openai.OpenAI(
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key=os.environ["ANYSCALE_API_KEY"],
    )

    # set up kwargs dict
    kwargs = {**default_decoding_kwargs, "max_tokens": L, "n": N, "logprobs": 1}
    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
    else:
        kwargs["temperature"] = 0

    # weird anyscale behaviors
    assert N == 1, "AnyScale only accepts n=1"
    if "llama-2-70b" in model.lower():
        kwargs.pop("logprobs")
    kwargs.pop("top_k")

    def _fmt_completion(d):
        outs = []
        for choice in d["choices"]:
            text = choice["text"] if "text" in choice else choice["message"]["content"]
            outs.append(
                APIOut(
                    full_completion=text,
                    L=d["usage"]["completion_tokens"],
                    num_prompt_tokens=d["usage"]["prompt_tokens"],
                )
            )
        return outs

    def _fmt_chat_completion(d):
        outs = []
        for choice in d["choices"]:
            logprobs = [l["logprob"] for l in choice["logprobs"]["content"]]
            outs.append(
                APIOut(
                    full_completion=choice["message"]["content"],
                    L=d["usage"]["completion_tokens"],
                    num_prompt_tokens=d["usage"]["prompt_tokens"],
                    completion_tokens=[
                        l["token"] for l in choice["logprobs"]["content"]
                    ],
                    completion_token_logprobs=logprobs,
                    full_logprob=np.sum(logprobs),
                )
            )
        return outs

    def get_fn(**kwargs):
        d = _call_openai_client(
            client,
            use_chat_endpoint=use_chat_endpoint,
            model=model,
            **kwargs,
        )
        try:
            if use_chat_endpoint:
                return _fmt_chat_completion(d)
            else:
                return _fmt_completion(d)
        except Exception as e:
            print(e)
            return d

    return get_fn, kwargs


def setup_together(
    model: str,
    N: int,
    L: int,
    use_chat_endpoint: bool = False,
    do_sample: bool = True,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
):
    HF_TO_TOGETHER = {
        "meta-llama/Meta-Llama-3-8B": "meta-llama/Llama-3-8b-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct": "meta-llama/Llama-3-8b-chat-hf",
        "meta-llama/Meta-Llama-3-70B-Instruct": "meta-llama/Llama-3-70b-chat-hf",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-405B-Instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    }
    if model in HF_TO_TOGETHER:
        model = HF_TO_TOGETHER[model]

    # set up kwargs dict
    kwargs = {
        **default_decoding_kwargs,
        "max_tokens": L,
        "n": N,
        "logprobs": 1,
        "echo": True,
    }
    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
    else:
        # from testing. conflicts with https://docs.together.ai/reference/chat-completions
        kwargs["temperature"] = 0

    def _fmt_completion(d):
        outs = []
        for i, choice in enumerate(d["choices"]):
            completion = choice["logprobs"]
            text = choice["text"] if "text" in choice else choice["message"]["content"]
            outs.append(
                APIOut(
                    prompt=(
                        d["prompt"][0]["text"].strip() if len(d["prompt"]) else None
                    ),
                    prompt_token_ids=d["prompt"][0]["logprobs"]["token_ids"],
                    full_completion=text,
                    completion_token_ids=completion.get("token_ids", None),
                    completion_tokens=completion.get("tokens", None),
                    completion_token_logprobs=completion.get("token_logprobs", None),
                    full_logprob=(
                        np.sum(completion["token_logprobs"])
                        if "token_logprobs" in completion
                        else None
                    ),
                    L=(
                        len(completion["tokens"])
                        if "tokens" in completion
                        else d["usage"]["completion_tokens"]
                    ),
                    num_prompt_tokens=d["usage"]["prompt_tokens"],
                )
            )
        return outs

    def _fmt_chat_completion(d):
        outs = []
        for i, choice in enumerate(d["choices"]):
            outs.append(
                APIOut(
                    prompt=(
                        d["prompt"][0]["text"].strip() if len(d["prompt"]) else None
                    ),
                    prompt_token_ids=d["prompt"][0]["logprobs"]["token_ids"],
                    full_completion=choice["message"]["content"],
                    completion_token_ids=choice["logprobs"]["token_ids"],
                    completion_tokens=choice["logprobs"]["tokens"],
                    completion_token_logprobs=choice["logprobs"]["token_logprobs"],
                    full_logprob=np.sum(choice["logprobs"]["token_logprobs"]),
                    L=len(choice["logprobs"]["tokens"]),
                    num_prompt_tokens=d["usage"]["prompt_tokens"],
                )
            )
        return outs

    def get_fn(prompt=None, system_message=None, **kwargs):
        if use_chat_endpoint:
            dialog = (
                [{"role": "system", "content": system_message}]
                if system_message is not None
                else []
            )
            dialog += [{"role": "user", "content": prompt}]
            payload = {"model": model, "messages": dialog, **kwargs}
            d = _call_requests(
                "https://api.together.xyz/v1/chat/completions",
                os.environ["TOGETHER_API_KEY"],
                payload,
            )
            try:
                return _fmt_chat_completion(d)
            except Exception as e:
                print(e, d)
                return d
        else:
            payload = {"model": model, "prompt": prompt, **kwargs}
            d = _call_requests(
                "https://api.together.xyz/v1/completions",
                os.environ["TOGETHER_API_KEY"],
                payload,
            )
            try:
                return _fmt_completion(d)
            except Exception as e:
                print(e, d)
                return d

    return get_fn, kwargs


def setup_fireworks(
    model: str,
    N: int,
    L: int,
    use_chat_endpoint: bool = False,
    do_sample: bool = True,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
):
    HF_TO_FIREWORKS_NAME = {
        "meta-llama/Llama-2-7b-hf": "accounts/fireworks/models/llama-v2-7b",
        "meta-llama/Llama-2-7b-chat-hf": "accounts/fireworks/models/llama-v2-7b-chat",
        "meta-llama/Llama-2-13b-chat-hf": "accounts/fireworks/models/llama-v2-13b-chat",
        "meta-llama/Llama-2-70b-chat-hf": "accounts/fireworks/models/llama-v2-70b-chat",
        "meta-llama/Meta-Llama-3-8B": "accounts/fireworks/models/llama-v3-8b-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct": "accounts/fireworks/models/llama-v3-8b-instruct-hf",
        "meta-llama/Meta-Llama-3-70B-Instruct": "accounts/fireworks/models/llama-v3-70b-instruct-hf",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "mistralai/Mistral-7B-v0.1": "accounts/fireworks/models/mistral-7b",
        "mistralai/Mistral-7B-Instruct-v0.1": "accounts/fireworks/models/mistral-7b-instruct-4k",
        "mistralai/Mistral-7B-Instruct-v0.2": "accounts/fireworks/models/mistral-7b-instruct-v0p2",
        "mistralai/Mixtral-8x7B-v0.1": "accounts/fireworks/models/mixtral-8x7b",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "accounts/fireworks/models/mixtral-8x7b-instruct",
        "meta-llama/Meta-Llama-3.1-405B-Instruct": "accounts/fireworks/models/llama-v3p1-405b-instruct",
    }
    model = HF_TO_FIREWORKS_NAME[model]

    # set up kwargs dict
    kwargs = {**default_decoding_kwargs, "max_tokens": L}
    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
    else:
        kwargs["temperature"] = 0
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def _fmt_completion(d):
        outs = []
        num_choices = len(d["choices"])
        for choice in d["choices"]:
            n = d["usage"]["completion_tokens"] // num_choices
            completion = [c for c in choice["logprobs"]["tokens"][-n:]]
            token_ids = [c for c in choice["logprobs"]["token_ids"][-n:]]
            logprobs = [c for c in choice["logprobs"]["token_logprobs"][-n:]]
            outs.append(
                APIOut(
                    full_completion="".join(completion),
                    completion_token_ids=token_ids,
                    completion_tokens=completion,
                    completion_token_logprobs=logprobs,
                    full_logprob=np.sum(logprobs),
                    num_prompt_tokens=d["usage"]["prompt_tokens"],
                    L=n,
                )
            )
        return outs

    def _fmt_chat_completion(d):
        outs = []
        num_choices = len(d["choices"])
        for choice in d["choices"]:
            n = d["usage"]["completion_tokens"] // num_choices
            outs.append(
                APIOut(
                    full_completion=choice["message"]["content"],
                    num_prompt_tokens=d["usage"]["prompt_tokens"],
                    L=n,
                )
            )
        return outs

    def get_fn(prompt=None, system_message=None, **kwargs):
        if use_chat_endpoint:
            dialog = (
                [{"role": "system", "content": system_message}]
                if system_message is not None
                else []
            )
            dialog += [{"role": "user", "content": prompt}]
            d = _call_requests(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                os.environ["FIREWORKS_API_KEY"],
                {"model": model, "messages": dialog, **kwargs},
            )
            try:
                return _fmt_chat_completion(d)
            except Exception as e:
                print(e, d)
                return d
        else:
            d = _call_requests(
                "https://api.fireworks.ai/inference/v1/completions",
                os.environ["FIREWORKS_API_KEY"],
                {"model": model, "prompt": prompt, **kwargs},
            )
            try:
                return _fmt_completion(d)
            except Exception as e:
                print(e, d)
                return d

    return get_fn, kwargs


def setup_perplexity(
    model: str,
    N: int,
    L: int,
    use_chat_endpoint: bool = False,
    do_sample: bool = True,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
):
    assert use_chat_endpoint, "Perplexity only supports chat completion endpoint"
    HF_TO_PPLX = {
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "mixtral-8x7b-instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct": "llama-3-8b-instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct": "llama-3-70b-instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama-3.1-8b-instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct": "llama-3.1-70b-instruct",
    }
    model = HF_TO_PPLX[model]
    client = openai.OpenAI(
        base_url=f"https://api.perplexity.ai",
        api_key=os.environ["PERPLEXITY_API_KEY"],
    )

    kwargs = {"max_tokens": L, "n": N}
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    else:
        kwargs["temperature"] = 0

    # weird perplexity behaviors
    assert N == 1, "Perplexity only supports n=1 samples."

    def get_fn(**kwargs):
        d = _call_openai_client(
            client, model=model, use_chat_endpoint=use_chat_endpoint, **kwargs
        )
        try:
            outs = []
            n = len(d["choices"])
            for choice in d["choices"]:
                outs.append(
                    APIOut(
                        full_completion=choice["message"]["content"],
                        num_prompt_tokens=d["usage"]["prompt_tokens"],
                        L=d["usage"]["completion_tokens"] / n,
                    )
                )
            return outs
        except Exception as e:
            print(e, d)
            return None

    return get_fn, kwargs


def setup_replicate(
    model: str,
    N: int,
    L: int,
    use_chat_endpoint: bool = False,
    do_sample: bool = True,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
):
    assert not use_chat_endpoint, "Replicate only supports completion endpoint"
    HF_TO_REPLICATE = {
        "meta-llama/Llama-2-7b-chat-hf": "meta/llama-2-7b-chat",
        "meta-llama/Llama-2-13b-chat-hf": "meta/llama-2-13b-chat",
        "meta-llama/Llama-2-70b-chat-hf": "meta/llama-2-70b-chat",
        "meta-llama/Meta-Llama-3-8B": "meta/meta-llama-3-8b",
        "meta-llama/Meta-Llama-3-8B-Instruct": "meta/meta-llama-3-8b-instruct",
        "meta-llama/Meta-Llama-3-70B": "meta/meta-llama-3-70b",
        "meta-llama/Meta-Llama-3-70B-Instruct": "meta/meta-llama-3-70b-instruct",
        "mistralai/Mistral-7B-v0.1": "mistralai/mistral-7b-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2": "mistralai/mistral-7b-instruct-v0.2",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistralai/mixtral-8x7b-instruct-v0.1",
        "meta-llama/Meta-Llama-3.1-405B-Instruct": "meta/meta-llama-3.1-405b-instruct",
    }
    model = HF_TO_REPLICATE[model]

    # set up kwargs dict
    kwargs = {**default_decoding_kwargs, "max_tokens": L, "n": N}
    assert N == 1, "Replicate only supports 1 sample"
    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
    else:
        kwargs["temperature"] = 0
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def get_fn(prompt=None, system_message=None, **kwargs):
        payload = {
            "input": {
                "prompt": prompt,
                "prompt_template": "{prompt}",
                **kwargs,
            }
        }
        if system_message is not None:
            payload["input"]["system_prompt"] = system_message
        first_json = _call_requests(
            f"https://api.replicate.com/v1/models/{model}/predictions",
            os.environ["REPLICATE_API_KEY"],
            payload,
        )
        get_url = first_json["urls"]["get"]
        d = None
        for i in range(50):  # MAX TRIES = 10
            second_json = _call_requests(
                get_url, os.environ["REPLICATE_API_KEY"], post=False
            )
            if second_json["status"] == "succeeded":
                d = second_json
                break
            else:
                time.sleep(1)
                get_url = second_json["urls"]["get"]
        if d is None:
            raise ValueError(f"Replicate API call failed; {second_json}")

        try:
            pattern = r"Formatted prompt: `([^`]*)`(?:(?!Formatted prompt: `).)*$"
            matches = re.findall(pattern, d["logs"], re.MULTILINE)
            if matches:
                prompt = matches[-1]
            else:
                prompt = None
            if len(d["metrics"]):
                input_token_count = d["metrics"][
                    "input_token_count"
                ]  # really confused by this, it might be that they added special [INST] tokens
            else:
                input_token_count = None
            return [
                APIOut(
                    prompt=prompt,
                    full_completion="".join(d["output"]),
                    completion_tokens=d["output"],
                    L=len(d["output"]),
                    num_prompt_tokens=input_token_count,
                )
            ]
        except Exception as e:
            print(e, d)
            return None

    return get_fn, kwargs


def setup_groq(
    model: str,
    N: int,
    L: int,
    use_chat_endpoint: bool = False,
    do_sample: bool = True,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
):
    assert use_chat_endpoint, "Groq only supports chat completions endpoint"
    HF_TO_GROQ = {
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama-3.1-8b-instant",
        "meta-llama/Meta-Llama-3.1-70B-Instruct": "llama-3.1-70b-versatile",
        "meta-llama/Meta-Llama-3-8B-Instruct": "llama3-8b-8192",
        "meta-llama/Meta-Llama-3-70B-Instruct": "llama3-70b-8192",
        "google/gemma-7b-it": "gemma-7b-it",
    }
    model = HF_TO_GROQ[model]
    assert N == 1, "Groq only supports n=1 samples"
    client = openai.OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    # set up kwargs dict
    kwargs = {**default_decoding_kwargs, "max_tokens": L, "n": N}
    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
    else:
        kwargs["temperature"] = 0

    # groq things
    kwargs.pop("top_k")

    def get_fn(**kwargs):
        d = _call_openai_client(
            client, model=model, use_chat_endpoint=use_chat_endpoint, **kwargs
        )
        try:
            return [
                APIOut(
                    full_completion=d["choices"][0]["message"]["content"],
                    L=d["usage"]["completion_tokens"],
                    num_prompt_tokens=d["usage"]["prompt_tokens"],
                )
            ]
        except Exception as e:
            print(e, d)
            return None

    return get_fn, kwargs


def setup_deepinfra(
    model: str,
    N: int,
    L: int,
    use_chat_endpoint: bool = False,
    do_sample: bool = True,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
):
    assert N == 1, "DeepInfra only support n = 1 samples"
    kwargs = {**default_decoding_kwargs, "max_tokens": L, "n": N}
    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
    else:
        kwargs["temperature"] = 0
    kwargs.pop("top_k")

    def _fmt_completion(d):
        outs = []
        n = len(d["choices"])
        for choice in d["choices"]:
            text = choice["text"] if "text" in choice else choice["message"]["content"]
            outs.append(
                APIOut(
                    full_completion=text,
                    L=d["usage"]["completion_tokens"] / n,
                    num_prompt_tokens=d["usage"]["prompt_tokens"],
                )
            )
        return outs

    def _fmt_chat_completion(d):
        outs = []
        n = len(d["choices"])
        for choice in d["choices"]:
            outs.append(
                APIOut(
                    full_completion=choice["message"]["content"],
                    L=d["usage"]["completion_tokens"] / n,
                    num_prompt_tokens=d["usage"]["prompt_tokens"],
                )
            )
        return outs

    def get_fn(**kwargs):
        client = openai.OpenAI(
            api_key=os.environ["DEEPINFRA_API_KEY"],
            base_url="https://api.deepinfra.com/v1/openai",
        )
        d = _call_openai_client(
            client, model=model, use_chat_endpoint=use_chat_endpoint, **kwargs
        )
        try:
            if use_chat_endpoint:
                return _fmt_chat_completion(d)
            else:
                return _fmt_completion(d)
        except Exception as e:
            print(e, d)
            return None

    return get_fn, kwargs


def setup_amazon(
    model: str,
    N: int,
    L: int,
    use_chat_endpoint: bool = False,
    do_sample: bool = True,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
):
    import boto3

    HF_TO_AMAZON = {
        "meta-llama/Meta-Llama-3-8B-Instruct": "meta.llama3-8b-instruct-v1:0",
        "meta-llama/Meta-Llama-3-70B-Instruct": "meta.llama3-70b-instruct-v1:0",
        "mistralai/Mistral-7B-Instruct-v0.2": "mistral.mistral-7b-instruct-v0:2",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "meta.llama3-1-8b-instruct-v1:0",
        "meta-llama/Meta-Llama-3.1-70B-Instruct": "meta.llama3-1-70b-instruct-v1:0",
        "meta-llama/Meta-Llama-3.1-405B-Instruct": "meta.llama3-1-405b-instruct-v1:0",
    }
    model = HF_TO_AMAZON[model]
    assert N == 1, "Amazon only supports 1 sample"
    kwargs = {
        "temperature": default_decoding_kwargs["temperature"],
        "top_p": default_decoding_kwargs["top_p"],
    }
    if "meta" in model:
        kwargs["max_gen_len"] = L
        assert top_k is None, "top_k unsupported for Amazon x Llama"
    elif "mistral" in model:
        kwargs["max_tokens"] = L
        if top_k is not None:
            kwargs["top_k"] = top_k

    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
    else:
        kwargs["temperature"] = 0

    client = boto3.client(service_name="bedrock-runtime")

    def _fmt_chat(d):
        return [
            APIOut(
                full_completion=d["output"]["message"]["content"][0]["text"],
                num_prompt_tokens=d["usage"]["inputTokens"],
                L=d["usage"]["outputTokens"],
                latency_ms=d["metrics"]["latencyMs"],
            )
        ]

    def _fmt_llama(d):
        return [
            APIOut(
                full_completion=d["generation"],
                L=d["generation_token_count"],
                num_prompt_tokens=d["prompt_token_count"],
            )
        ]

    def _fmt_mistral(d):
        outs = []
        for o in d["outputs"]:
            outs.append(
                APIOut(
                    full_completion=o["text"],
                )
            )
        return outs

    def get_fn(prompt=None, system_message=None, **kwargs):
        if use_chat_endpoint:
            messages = (
                [{"role": "system", "content": [{"text": system_message}]}]
                if system_message is not None
                else []
            )
            messages += [{"role": "user", "content": [{"text": prompt}]}]
            d = client.converse(
                modelId=model,
                messages=messages,
                inferenceConfig={
                    k: v
                    for k, v in kwargs.items()
                    if k in ["maxTokens", "stopSequences", "temperature", "topP"]
                },
                additionalModelRequestFields={
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["maxTokens", "stopSequences", "temperature", "topP"]
                },
            )
            fmt_fn = _fmt_chat
        else:
            body = json.dumps(
                {
                    "prompt": prompt,
                    **kwargs,
                }
            )
            accept = "application/json"
            contentType = "application/json"
            response = client.invoke_model(
                body=body, modelId=model, accept=accept, contentType=contentType
            )
            d = json.loads(response.get("body").read())
            if "meta" in model:
                fmt_fn = _fmt_llama
            elif "mistral" in model:
                fmt_fn = _fmt_mistral
        # format output
        try:
            return fmt_fn(d)
        except Exception as e:
            print(e, d)
            return None

    return get_fn, kwargs


def setup_azure(
    model: str,
    N: int,
    L: int,
    use_chat_endpoint: bool = False,
    do_sample: bool = True,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
):
    HF_TO_AZURE = {
        "meta-llama/Meta-Llama-3-8B-Instruct": "https://Meta-Llama-3-8B-Instruct-jylwk.eastus2.models.ai.azure.com/v1/chat/completions",
        "meta-llama/Meta-Llama-3-70B-Instruct": "https://Meta-Llama-3-70B-Instruct-lqyfw.eastus2.models.ai.azure.com/v1/chat/completions",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "https://Meta-Llama-3-1-8B-Instruct-mcjym.eastus2.models.ai.azure.com/v1/chat/completions",
        "meta-llama/Meta-Llama-3.1-70B-Instruct": "https://Meta-Llama-3-1-70B-Instruct-tmzr.eastus2.models.ai.azure.com/v1/chat/completions",
        "meta-llama/Meta-Llama-3.1-405B-Instruct": "https://Meta-Llama-3-1-405B-Instruct-tqx.eastus2.models.ai.azure.com/v1/chat/completions",
    }
    url = HF_TO_AZURE[model]
    key = json.load(open("/sailhome/irena/.ssh/azure_key.json"))[model]
    assert N == 1
    assert use_chat_endpoint, "Azure only supports chat completions endpoint"
    kwargs = {
        "temperature": default_decoding_kwargs["temperature"],
        "top_p": default_decoding_kwargs["top_p"],
        "max_tokens": L,
        "echo": True,
    }
    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
    else:
        kwargs["temperature"] = 0

    def fmt_fn(d):
        out = d["choices"][0]
        return [
            APIOut(
                full_completion=out["message"]["content"],
                num_prompt_tokens=d["usage"]["prompt_tokens"],
                L=d["usage"]["completion_tokens"],
            )
        ]

    def get_fn(prompt=None, system_message=None, **kwargs):
        dialog = (
            [{"role": "system", "content": system_message}]
            if system_message is not None
            else []
        )
        dialog += [{"role": "user", "content": prompt}]
        payload = {"model": model, "messages": dialog, **kwargs}
        d = _call_requests(url, key, payload)
        try:
            return fmt_fn(d)
        except Exception as e:
            print(e, d)
            return None

    return get_fn, kwargs


def setup_openai(
    model: str,
    N: int,
    L: int,
    use_chat_endpoint: bool = False,
    do_sample: bool = True,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
):
    kwargs = {
        "temperature": default_decoding_kwargs["temperature"],
        "top_p": default_decoding_kwargs["top_p"],
        "max_tokens": L,
    }
    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
    else:
        kwargs["temperature"] = 0

    import openai

    client = openai.OpenAI()

    def fmt_fn(d):
        out = d["choices"][0]
        return [
            APIOut(
                full_completion=out["message"]["content"],
                num_prompt_tokens=d["usage"]["prompt_tokens"],
                L=d["usage"]["completion_tokens"],
            )
        ]

    def get_fn(prompt=None, system_message=None, **kwargs):
        d = _call_openai_client(
            client=client,
            model=model,
            prompt=prompt,
            system_message=system_message,
            use_chat_endpoint=True,
            **kwargs,
        )
        try:
            return fmt_fn(d)
        except Exception as e:
            print(e, d)
            return None

    return get_fn, kwargs


###################################################


def _call_openai_client(
    client,
    *,
    model,
    prompt: str,
    system_message: str = None,
    use_chat_endpoint=False,
    **kwargs,
) -> dict:
    if use_chat_endpoint:
        dialog = (
            [{"role": "system", "content": system_message}]
            if system_message is not None
            else []
        )
        dialog += [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model=model, messages=dialog, **kwargs
        )
    else:
        assert system_message is None
        completion = client.completions.create(model=model, prompt=prompt, **kwargs)
    return completion.model_dump()


def _call_requests(url, key, payload=None, post=True) -> dict:
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    if post:
        assert payload is not None
        response = requests.post(url, headers=headers, json=payload)
    else:
        response = requests.get(url, headers=headers)
    try:
        return response.json()
    except Exception as e:
        print(e, response)
        return None
