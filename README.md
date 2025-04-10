# LLM API Audit

This project provides different methods for auditing Large Language Models (LLMs) to verify service integrity.

📄 **Paper**: [Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs](https://arxiv.org/abs/2504.04715)

## Methods

### Classifier
Adapted from [LLM Idiosyncrasies](https://github.com/locuslab/llm-idiosyncrasies/tree/main) with added model support.

1. Generate responses from all models mentioned in paper:
   ```bash
   ./run_all.sh
   ```

2. Train binary classifiers between original and quantized models. This example uses LLM2Vec, but other embedding models can be easily added:
   ```bash
   ./classify.sh
   ```

3. Analyze classification results in the `./classification`.

---

### Identity Prompting

1. Generate identity responses for multiple models:
   ```bash
   ./run_generation.sh
   ```

2. Analyze identity occurences using the example shown in the notebook:
   ```bash
   jupyter notebook count_names.ipynb
   ```

---

### Model Equality Testing
Adapted from [Model Equality Testing](https://github.com/i-gao/model-equality-testing/tree/main) with randomized model substitution and relevant experiments.

1. Download the necessary datasets that includes selected LLMs generation on wikipedia dataset:
   ```bash
   cd model_equality_testing
   python download.py
   ```

2. Run model equality testing on mixed distribution with different probability of substitution, parameters for the experiments can also be changed here:
   ```bash
   ./mixed.sh
   ```

3. Results are saved in model-specific directories with summaries showing statistical power of the test.

---

### Benchmark
Adapted from [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) with added temperature support token loglikelihood request.

1. Set up the benchmark environment:
   ```bash
   cd benchmark
   pip install -e .
   ```

2. Run different benchmarks at different temperatures:
   ```bash
   ./run_benchmarks.sh
   ```

3. MMLU requires an extra step, resampling MMLU results for Monte Carlo estimation:
   ```bash
   python resample_mmlu.py --dir "/path/to/model/mmlu_results" --samples 100
   ```

---

### Logprobs

1. Collect logprobs from models. Use `pip install` to specify different versions of `vllm` and `transformers` to vary the software environment:
   ```bash
   python logprobs/run_logprobs.py --model "meta-llama/Meta-Llama-3-8B-Instruct"
   ```

---

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{cai2025gettingpayforauditing,
   title={Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs}, 
   author={Cai, Will and Shi, Tianneng and Zhao, Xuandong and Song, Dawn},
   journal={arXiv preprint arXiv:2504.04715},
   year={2025},
}
```


---

*Note: As of April 8, 2025, we are still cleaning up the code and rerunning experiments to ensure everything works as expected. We will make necessary modifications and provide more details about the environment soon.*

