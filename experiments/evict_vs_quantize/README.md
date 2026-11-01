# Evict or quantize?

For long inputs, how do **H2O-style KV-cache eviction** (dropping tokens) and **KV-cache quantisation** (storing them in fewer bits) compare, do they combine, and does the answer change by task? Costs are measured on the GPU (latency and energy), not estimated from FLOPs.

> **Status:** code written and syntax-checked; scoring and summary tested on dummy data. Not yet run on a GPU.

## Design

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` in fp16 with eager attention (28 layers, 2 KV heads, head size 128, so 28.7 KB of KV cache per token).

**Tasks:** RULER at 4,096 tokens ([`simonjegou/ruler`](https://huggingface.co/datasets/simonjegou/ruler)), 25 questions each, fixed seed.

| Task | Tests |
|---|---|
| `niah_single_1` | Retrieving one hidden fact |
| `vt` | Following chained variable assignments (multi-hop) |
| `qa_1` | Answering from real documents with distractors |

**Conditions:**

| Condition | Tokens evicted | KV bits | KV cache (whole model, 4k tokens) |
|---|---|---|---|
| `full_fp16` | 0% | 16 | 117.4 MB |
| `h2o_50` | 50% | 16 | 58.7 MB |
| `h2o_75` | 75% | 16 | 29.4 MB |
| `kv_int4` | 0% | 4 | 29.4 MB |
| `kv_int2` | 0% | 2 | 14.7 MB |
| `h2o_50+kv_int4` | 50% | 4 | 14.7 MB |

Two equal-memory pairs (`h2o_75` vs `kv_int4`, and `kv_int2` vs `h2o_50+kv_int4`) answer the headline question: at the same budget, drop tokens or lower precision?

**Measured per answer:** score (RULER exact-answer match), latency, GPU energy (NVML), peak GPU memory and calculated KV-cache size.

## Run on Kaggle

1. Verify your phone number on Kaggle (needed for GPU and internet).
2. New notebook, then Settings: Accelerator **GPU T4 x2**, Internet **On**.
3. Run:

```python
!git clone https://github.com/seshjuansauce/kv-cache-compression-bench.git
%cd kv-cache-compression-bench
!pip install -q -r requirements.txt
%cd experiments/evict_vs_quantize

# smoke test (about 5-10 minutes)
!CUDA_VISIBLE_DEVICES=0 python kv_eviction_vs_quant.py --n 2 --out smoke.csv

# full run: 6 conditions x 75 questions = 450 answers (about 1.5-2.5 hours)
!CUDA_VISIBLE_DEVICES=0 python kv_eviction_vs_quant.py --n 25
!python kv_eviction_vs_quant.py --summarise
```

For the full run, use **Save Version > Save & Run All (Commit)** so it runs in the background; files in `/kaggle/working` are lost when an interactive session ends. Download the outputs from the version's Output tab.

The repo must be public for the `git clone` above to work without a token.

## Outputs

| File | Contents |
|---|---|
| `results.csv` | One row per answer: condition, task, score, latency, energy, peak memory, tokens, KV size, prediction, references |
| `summary.csv` | Accuracy per task and condition, plus mean latency, energy and KV size |
| `accuracy_vs_kv.png` | Accuracy against KV size, one panel per task |

## Results

*To be added after the run.*

## Limitations

- The eviction is **H2O-style** (kvpress `ObservedAttentionPress`): it compresses once, after reading the context and before the question, with no recent-token window. The original H2O evicts continuously during generation.
- KV sizes are calculated from the model config and ignore quantisation scale overhead.
- With 25 questions per task, differences under about 20 points are within noise.
- One model, one context length. At 4k tokens the cache is small next to the weights, so little latency gain is expected.

## Planned extension

Question-aware eviction: place the question after the context and score tokens only by the question's attention (the SnapKV idea), to test whether the needle is lost because the press scores before the question exists. Details in the [documentation PDF](../../docs/kv_eviction_vs_quant_documentation.pdf).
