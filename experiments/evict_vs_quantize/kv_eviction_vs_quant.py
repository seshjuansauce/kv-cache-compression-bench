"""
kv_eviction_vs_quant.py

Question: for long inputs, how do H2O-style KV-cache eviction (Beidi Chen's group,
NeurIPS 2023) and KV-cache quantisation compare, and do they combine, task by task?
Costs are measured the way Strubell et al. (ACL 2025) recommend: real latency and
energy on the GPU, not FLOP estimates.

Conditions (all on the same model, eager attention, fp16):
  full_fp16        full KV cache, no compression (baseline)
  h2o_50           drop 50% of context KV pairs by observed attention (H2O-style)
  h2o_75           drop 75%
  kv_int4          keep every token, store the KV cache in 4-bit (quanto)
  kv_int2          keep every token, store the KV cache in 2-bit
  h2o_50+kv_int4   drop 50%, store the rest in 4-bit

Tasks (RULER, 4k-token contexts, from kvpress's preprocessed copy):
  niah_single_1    retrieve one fact hidden in a long text
  vt               variable tracking (follow a chain of assignments)
  qa_1             question answering over a long document

Per example it records: score, latency, peak GPU memory, GPU energy (NVML),
and an analytic KV-cache size for the context.

Runs on one NVIDIA GPU. A free Kaggle or Colab T4 is enough.

  pip install -q kvpress optimum-quanto nvidia-ml-py datasets pandas matplotlib
  python kv_eviction_vs_quant.py --n 25
  python kv_eviction_vs_quant.py --summarise   # tables + plot from results.csv

Results are appended to results.csv after every example, so a dropped session
can be resumed by running the same command again.
"""

import argparse
import os
import threading
import time

import pandas as pd
import torch
import warnings

# One question at a time is deliberate (clean per-answer latency and energy), so silence
# transformers' hint to batch pipeline calls through a dataset.
warnings.filterwarnings("ignore", message="You seem to be using the pipelines sequentially on GPU")

CONDITIONS = {
    "full_fp16": dict(ratio=0.0, nbits=None),
    "h2o_50": dict(ratio=0.5, nbits=None),
    "h2o_75": dict(ratio=0.75, nbits=None),
    "kv_int4": dict(ratio=0.0, nbits=4),
    "kv_int2": dict(ratio=0.0, nbits=2),
    "h2o_50+kv_int4": dict(ratio=0.5, nbits=4),
}
DEFAULT_TASKS = ["niah_single_1", "vt", "qa_1"]


# ---------------------------------------------------------------- energy
class EnergyMeter:
    """GPU energy in joules over a block. Uses the NVML energy counter when the
    GPU supports it, otherwise integrates sampled power."""

    def __init__(self, index=0, interval=0.05):
        self.ok, self.counter, self.interval = False, False, interval
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.ok = True
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                self.counter = True
            except Exception:
                self.counter = False
        except Exception as e:  # no NVML: energy is reported as NaN
            print(f"[energy] NVML unavailable ({e}); energy will be NaN")

    def __enter__(self):
        self.joules = float("nan")
        if not self.ok:
            return self
        if self.counter:
            self.e0 = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)  # mJ
        else:
            self.samples, self._stop = [], False

            def run():
                while not self._stop:
                    watts = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                    self.samples.append((time.perf_counter(), watts))
                    time.sleep(self.interval)

            self.thread = threading.Thread(target=run, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        if not self.ok:
            return False
        if self.counter:
            e1 = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.joules = (e1 - self.e0) / 1000.0
        else:
            self._stop = True
            self.thread.join()
            s = self.samples
            self.joules = sum((t2 - t1) * (p1 + p2) / 2 for (t1, p1), (t2, p2) in zip(s, s[1:]))
        return False


# ---------------------------------------------------------------- scoring
def score_example(task, pred, refs):
    """Same rule as kvpress's RULER metric: qa tasks get credit if any reference
    appears in the prediction; other tasks get the fraction of references found."""
    pred = pred.lower()
    hits = [1.0 if str(r).lower() in pred else 0.0 for r in refs]
    if task.split("_")[0] == "qa":
        return 100.0 * max(hits)
    return 100.0 * sum(hits) / len(hits)


# ---------------------------------------------------------------- model + caches
def load_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    # Eager attention is required by kvpress's ObservedAttentionPress (H2O-style).
    # It is used for every condition so attention cost is the same everywhere.
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, attn_implementation="eager"
    )
    if model.dtype != torch.float16:
        model = model.half()
    model = model.to("cuda").eval()
    return model, tok


def make_cache(nbits):
    """A fresh quantised KV cache, or None for the default full-precision cache.
    Handles both the newer and older transformers APIs."""
    if nbits is None:
        return None
    try:
        from transformers import QuantizedCache

        return QuantizedCache(backend="quanto", nbits=nbits)
    except TypeError:
        from transformers import QuantizedCacheConfig, QuantoQuantizedCache

        return QuantoQuantizedCache(cache_config=QuantizedCacheConfig(nbits=nbits))


def make_press(ratio):
    if ratio <= 0:
        return None
    from kvpress import ObservedAttentionPress

    return ObservedAttentionPress(compression_ratio=ratio)


def kv_bytes_per_token(model):
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return 2 * n_layers * n_kv * head_dim * 2  # keys + values, fp16 = 2 bytes



# ---------------------------------------------------------------- Hugging Face token
def setup_hf_token():
    """Use HF_TOKEN from the environment, or from Kaggle Secrets when running on Kaggle.
    The token is never hard-coded or printed. Without it, downloads still work but are
    unauthenticated (slower and rate-limited)."""
    if os.environ.get("HF_TOKEN"):
        print("[hf] using HF_TOKEN from the environment")
        return
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            print("[hf] using HF_TOKEN from Kaggle Secrets")
            return
    except Exception:
        pass
    print("[hf] no HF_TOKEN found; downloading without authentication")

# ---------------------------------------------------------------- run
def run(args):
    setup_hf_token()  # before any download
    from datasets import load_dataset
    from transformers import pipeline
    import kvpress  # noqa: F401  importing kvpress registers the "kv-press-text-generation" pipeline

    assert torch.cuda.is_available(), "Needs an NVIDIA GPU (Kaggle/Colab T4 is fine)."

    df = load_dataset("simonjegou/ruler", args.config, split="test").to_pandas()
    available = sorted(df["task"].unique())
    missing = [t for t in args.tasks if t not in available]
    if missing:
        raise SystemExit(f"Unknown tasks {missing}. Available: {available}")
    sample = pd.concat(
        df[df["task"] == t].sample(n=min(args.n, (df["task"] == t).sum()), random_state=0)
        for t in args.tasks
    ).reset_index(names="row_id")

    model, tok = load_model(args.model)
    pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tok, device=0)
    per_token = kv_bytes_per_token(model)
    meter = EnergyMeter()

    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = set(zip(prev["condition"], prev["row_id"]))
        print(f"[resume] {len(done)} rows already in {args.out}")

    conditions = args.conditions or list(CONDITIONS)
    for cond in conditions:
        spec = CONDITIONS[cond]
        todo = sample[[(cond, r) not in done for r in sample["row_id"]]]
        if todo.empty:
            continue
        print(f"\n=== {cond}: {len(todo)} examples ===")

        # warm-up so the first timed example is not penalised
        try:
            w = todo.iloc[0]
            pipe(w["context"], question=w["question"], answer_prefix=w["answer_prefix"],
                 press=make_press(spec["ratio"]), cache=make_cache(spec["nbits"]), max_new_tokens=8)
        except Exception as e:
            print(f"[skip] {cond} failed on warm-up: {type(e).__name__}: {e}")
            continue

        for _, row in todo.iterrows():
            n_ctx = len(tok(row["context"]).input_ids)
            press, cache = make_press(spec["ratio"]), make_cache(spec["nbits"])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with meter:
                out = pipe(row["context"], question=row["question"],
                           answer_prefix=row["answer_prefix"], press=press, cache=cache,
                           max_new_tokens=int(row["max_new_tokens"]))
                torch.cuda.synchronize()
            latency = time.perf_counter() - t0
            pred = out.get("answer") or (out.get("answers") or [""])[0]
            refs = list(row["answer"])
            bits = spec["nbits"] or 16
            rec = dict(
                condition=cond, row_id=row["row_id"], task=row["task"],
                score=score_example(row["task"], pred, refs),
                latency_s=round(latency, 3),
                energy_j=round(meter.joules, 2),
                peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3),
                ctx_tokens=n_ctx,
                kv_ctx_mb=round(n_ctx * (1 - spec["ratio"]) * per_token * bits / 16 / 1e6, 1),
                pred=pred[:200].replace("\n", " "), refs="|".join(map(str, refs)),
            )
            pd.DataFrame([rec]).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
            print(f"{cond:16s} {row['task']:14s} score={rec['score']:5.1f} "
                  f"{latency:5.1f}s {rec['energy_j']:7.1f}J kv={rec['kv_ctx_mb']}MB")


# ---------------------------------------------------------------- summary
def summarise(args):
    if not os.path.exists(args.out):
        raise SystemExit(f"{args.out} not found: the run wrote no results. Check the run's output above for the first error.")
    df = pd.read_csv(args.out)
    order = [c for c in CONDITIONS if c in df["condition"].unique()]

    acc = df.pivot_table(index="condition", columns="task", values="score", aggfunc="mean").loc[order]
    cost = df.groupby("condition").agg(
        kv_ctx_mb=("kv_ctx_mb", "mean"),
        latency_s=("latency_s", "mean"),
        energy_j=("energy_j", "mean"),
        peak_mem_gb=("peak_mem_gb", "max"),
        n=("score", "size"),
    ).loc[order]
    table = acc.round(1).join(cost.round(2))
    print(table.to_string())
    table.to_csv("summary.csv")

    import matplotlib.pyplot as plt

    tasks = list(acc.columns)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5 * len(tasks), 3.8), sharey=True)
    axes = [axes] if len(tasks) == 1 else axes
    for ax, task in zip(axes, tasks):
        for cond in order:
            ax.scatter(cost.loc[cond, "kv_ctx_mb"], acc.loc[cond, task], s=40)
            ax.annotate(cond, (cost.loc[cond, "kv_ctx_mb"], acc.loc[cond, task]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_title(task)
        ax.set_xlabel("KV cache for context (MB)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Score (%)")
    fig.tight_layout()
    fig.savefig("accuracy_vs_kv.png", dpi=160)
    print("\nWrote summary.csv and accuracy_vs_kv.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--config", default="4096", help="RULER context length: 4096, 8192 or 16384")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--n", type=int, default=25, help="examples per task")
    p.add_argument("--conditions", nargs="+", choices=list(CONDITIONS))
    p.add_argument("--out", default="results.csv")
    p.add_argument("--summarise", action="store_true")
    a = p.parse_args()
    summarise(a) if a.summarise else run(a)
