# kv-cache-compression-bench

Small, measured experiments on compressing the KV cache of language models: what each method saves in memory, latency and energy, and what it costs in accuracy, task by task.

## Experiments

| Folder | Question | Status |
|---|---|---|
| [`experiments/evict_vs_quantize`](experiments/evict_vs_quantize) | At the same memory budget, is it better to drop KV-cache tokens (H2O-style eviction) or store them in fewer bits (quantisation)? Does the answer change by task? | Code ready, not yet run |
| `experiments/kivi_reproduction` | Reproduce KIVI's 2-bit KV-cache quantisation on a small open model | Planned |

## Setup

```bash
pip install -r requirements.txt
```

The experiments need an NVIDIA GPU: energy is read from NVML and the quantised KV cache uses CUDA. A free Kaggle or Colab T4 is enough. Each experiment's README explains how to run it.

## Documentation

- [`docs/kv_eviction_vs_quant_documentation.pdf`](docs/kv_eviction_vs_quant_documentation.pdf): design, code architecture, how to run, mechanisms, expected results and limitations for `evict_vs_quantize`.

## Key references

- Zhang et al., [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048), NeurIPS 2023
- Liu et al., [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750), ICML 2024
- Fernandez et al., [Energy Considerations of Large Language Model Inference and Efficiency Optimizations](https://preview.aclanthology.org/setup/2025.acl-long.1563), ACL 2025
- Hsieh et al., [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654), 2024
- NVIDIA, [kvpress](https://github.com/NVIDIA/kvpress)
