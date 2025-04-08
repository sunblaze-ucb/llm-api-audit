# Installation

Please use two separate environments for response generation and classification. It is possible to merge them into a single environment, but it takes a lot of time to resolve the dependencies.

## Dependency Setup for Response Generation

```
conda create -n gen_resp python=3.9 -y
conda activate gen_resp
pip install vllm==0.6.3.post1 datasets==3.2.0 openai anthropic google-generativeai
```

## Dependency Setup for Classification

```
conda create -n classification python=3.9 -y
conda activate classification
pip install llm2vec==0.2.3 tensorboard
```