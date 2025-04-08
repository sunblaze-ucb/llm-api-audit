import sys
sys.path.append("../")
from accelerate import Accelerator

APIS = [
    "anyscale",
    "together",
    "fireworks",
    "perplexity",
    "replicate",
    "groq",
    "deepinfra",
    "amazon",
    "azure",
]
MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
]

############

from experiments.sampling.model import TransformersModel
import experiments.prompts as prompts
from cache_api_samples import policy
import experiments.sampling.api as api_library

def get_expected_prompt_len(api, model, ds): 
    _policy = policy(api, model.model_name)
    p = ds[0][_policy["prompt_key"]]
    p_for_len = ds[0][_policy["expected_prompt_key"]]
    ids = model.tokenizer(
        p_for_len, add_special_tokens=('special' not in _policy["expected_prompt_key"]),
    )['input_ids']
    return len(ids), p

def test_sampling(p, model, api, expected_prompt_len, n):
    print(f">> Testing {model} with {api} and repeatedly requesting through n={n}")
    i = 0
    try:
        raw = ""
        out = []
        get_fn, kwargs = getattr(api_library, f"setup_{api}")(
            model=model,
            N=1,
            L=5,
            use_chat_endpoint=policy(api, model)["use_chat_endpoint"],
            do_sample=True,
            temperature=1,
            top_p=None,
        )
        for i in range(n):
            o = get_fn(prompt=p, **kwargs)[0]
            if not expected_prompt_len == o.num_prompt_tokens:
                if o.prompt is None or o.prompt != p:
                    raise Exception(f"Expected prompt len was {expected_prompt_len}, actual was {o.num_prompt_tokens}\n{o}")
            out.append(o.full_completion)
        if len(set(out)) == 1:
            raise Exception("Always got the same output: " + out[0])
    except Exception as e:
        print("\tResult: FAILED")
        print("\tIteration " + str(i) + ": " + str(e) + str(raw))
        return False
    print("\tResult: PASSED")   
    return True

#############

accelerator = Accelerator()

for m in MODELS:
    model = TransformersModel(m, accelerator, skip_loading_weights=True)
    ds = prompts.get_bit_prompts(model)
    for api in APIS:
        expected_prompt_len, prompt = get_expected_prompt_len(api, model, ds)
        # n=1, repeated sampling for 3 times
        test_sampling(prompt, m, api, expected_prompt_len, 2)
