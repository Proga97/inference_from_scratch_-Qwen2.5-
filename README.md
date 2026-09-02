# Qwen2.5-0.5B Inference From Scratch

A from-scratch PyTorch implementation of **Qwen2.5-0.5B inference**, focused on understanding and measuring the mechanics of Transformer inference rather than hiding them behind high-level generation APIs.

The project starts by reproducing the pretrained model's forward pass, verifies numerical agreement with Hugging Face, then progressively builds a faster inference stack with **KV caching**, **sampling**, and a reproducible **benchmark harness**.

---

## Project Goals

This project is built around five experiments.

### 1. Reimplement Qwen inference

Load the pretrained **Qwen2.5-0.5B** weights into custom PyTorch modules.

No:

```python
model.generate()
```

No Hugging Face pipeline.

The forward pass is implemented directly, including:

```text
Token Embeddings
      ↓
RMSNorm
      ↓
Self-Attention
      ├── Q / K / V projections
      ├── Head reshaping
      ├── RoPE
      ├── Grouped Query Attention
      ├── Scaled dot-product attention
      ├── Causal masking
      └── Output projection
      ↓
Residual connection
      ↓
SwiGLU MLP
      ↓
Residual connection
      ↓
Final RMSNorm
      ↓
LM Head
      ↓
Logits
```

The custom implementation is verified against Hugging Face with a target maximum logit difference of approximately:

```text
≤ 1e-4
```

---

### 2. Naive greedy decoding

Implement autoregressive generation manually.

For every generated token, the entire sequence is passed through the model again:

```text
Prompt
  ↓
Full forward pass
  ↓
argmax
  ↓
append token
  ↓
Full forward pass again
  ↓
argmax
  ↓
...
```

The benchmark measures:

```text
Context length → Tokens / second
```

across increasing sequence lengths.

The expected behavior is a substantial reduction in throughput as context grows because self-attention repeatedly processes the full sequence.

---

### 3. KV cache

Add an incremental **Key/Value cache** to avoid recomputing previous tokens during decoding.

Instead of:

```text
Step N:
recompute K and V for tokens 0 ... N
```

the cached version does:

```text
Step N:
reuse K/V for tokens 0 ... N-1
compute K/V only for token N
```

The benchmark will run both implementations under the same conditions:

```text
Naive decoding
      vs
KV-cache decoding
```

The primary result is the before/after throughput graph:

```text
Tokens/sec
   │
   │       ───────── KV cache
   │      /
   │     /
   │    /
   │   /
   │  /
   │ /
   │────────────── Naive
   └──────────────────── Context length
```

The README will also compare measured KV-cache performance with the **theoretical memory-bandwidth ceiling** derived from the model's memory traffic.

The point is not simply that KV caching is faster; it is to understand **why** it is faster and what prevents real hardware from reaching the theoretical limit.

---

### 4. Sampling

Implement configurable token sampling without relying on Hugging Face generation utilities.

Supported controls:

```text
Temperature
Top-k
Top-p
Seeded RNG
```

The sampler will support deterministic reproduction through an explicit random seed.

Conceptually:

```text
logits
  ↓
temperature scaling
  ↓
top-k filtering
  ↓
top-p filtering
  ↓
softmax
  ↓
sample token
```

The project will include examples comparing:

```text
Greedy decoding
vs
Sampled decoding
```

using the same prompt.

Example:

```text
Greedy:
"The future of AI is ..."

Sampled:
"The future of AI may evolve through ..."
```

The exact outputs will depend on the model, prompt, sampling configuration, and seed.

---

### 5. Reproducible benchmark harness

All performance experiments are driven from:

```text
bench.py
```

The goal is for a fresh clone of the repository to be able to regenerate the graphs used throughout this README.

The harness will cover:

```text
Naive decoding vs context length
KV-cache decoding vs context length
Naive vs KV-cache throughput
Sampling benchmarks
```

Benchmark results should be saved alongside the generated plots so the experiments are reproducible rather than relying on manually captured numbers.

---

# Architecture

The custom implementation mirrors the Qwen decoder architecture and module hierarchy closely enough to load the pretrained checkpoint directly.

```text
Qwen2.5-0.5B
│
├── Token Embedding
│
├── Decoder Layer × N
│   │
│   ├── Input RMSNorm
│   │
│   ├── Self-Attention
│   │   ├── q_proj
│   │   ├── k_proj
│   │   ├── v_proj
│   │   ├── RoPE
│   │   ├── GQA
│   │   ├── Attention
│   │   └── o_proj
│   │
│   ├── Residual Connection
│   │
│   ├── Post-Attention RMSNorm
│   │
│   ├── SwiGLU MLP
│   │   ├── gate_proj
│   │   ├── up_proj
│   │   └── down_proj
│   │
│   └── Residual Connection
│
├── Final RMSNorm
│
└── LM Head
```

The implementation uses the model configuration to determine architecture dimensions instead of hardcoding them.

---

# Numerical Verification

The model is developed and verified incrementally.

```text
Embedding             ✅
RMSNorm               ✅
Q / K / V projections ✅
Head reshaping        ✅
RoPE                  ✅
GQA                   ✅
Attention             ✅
SwiGLU MLP            ✅
Decoder layer         ✅
Full model            ✅
Logits                ✅
```

The primary correctness test is:

```text
HF logits
    vs
Custom-model logits
```

with:

```python
max(abs(hf_logits - my_logits)) <= 1e-4
```

This provides a numerical check that the custom forward pass is reproducing the reference model rather than merely generating plausible text.

---

# Naive Decoding

The naive decoder intentionally performs a full forward pass for every generated token.

For context length `N`:

```text
Token 1 → forward(N)
Token 2 → forward(N + 1)
Token 3 → forward(N + 2)
...
```

This repeatedly recomputes work that does not change.

The benchmark demonstrates how this affects generation throughput as context grows.

Parity Test Results:

````text
hidden[ 0] max diff = 0.00e+00
hidden[ 1] max diff = 2.15e-06
hidden[ 2] max diff = 1.91e-06
hidden[ 3] max diff = 3.93e-06
hidden[ 4] max diff = 7.63e-06
hidden[ 5] max diff = 7.63e-06
hidden[ 6] max diff = 7.63e-06
hidden[ 7] max diff = 7.63e-06
hidden[ 8] max diff = 7.63e-06
hidden[ 9] max diff = 7.63e-06
hidden[10] max diff = 7.63e-06
hidden[11] max diff = 7.63e-06
hidden[12] max diff = 7.63e-06
hidden[13] max diff = 7.63e-06
hidden[14] max diff = 7.63e-06
hidden[15] max diff = 9.54e-06
hidden[16] max diff = 9.06e-06
hidden[17] max diff = 1.24e-05
hidden[18] max diff = 1.34e-05
hidden[19] max diff = 1.34e-05
hidden[20] max diff = 2.29e-05
hidden[21] max diff = 3.05e-05
hidden[22] max diff = 3.66e-04
hidden[23] max diff = 1.77e-04

HF logits shape: torch.Size([1, 6, 151936])
My logits shape: torch.Size([1, 6, 151936])

logits max diff = 3.4332275390625e-05

M1 parity: True

HF next token :  Washington
My next token :  Washington
````
---

# KV Cache

The KV cache changes the decoding computation from:

```text
Every step:
recompute all previous K/V
````

to:

```text
Every step:
reuse cached K/V
compute only the new token's K/V
```

This dramatically reduces the amount of computation performed during autoregressive decoding.

The implementation will cache the projected and position-encoded:

```text
Key
Value
```

states for each decoder layer.

During decoding, only the newly generated token is processed through the attention projections while cached states are reused.

---

# Why KV Cache Helps

Without caching, decoding repeatedly performs work proportional to the full context.

With caching:

```text
Previous K/V:
stored

New token:
Q, K, V computed

Attention:
new Q attends over
cached K/V + new K/V
```

The computational pattern therefore becomes much closer to **incremental decoding** rather than repeatedly running full-context inference.

---

# Theoretical Bandwidth Ceiling

KV-cache decoding is often discussed as a memory-bandwidth problem.

A simplified upper bound can be estimated from:

```text
model weight traffic
--------------------
memory bandwidth
```

and, during decode, from the amount of KV-cache and parameter data that must be moved through memory per generated token.

The measured throughput will generally remain below this idealized ceiling because real inference also pays for:

```text
Kernel launch overhead
Matrix multiplication efficiency
Memory access patterns
Synchronization
Tensor layout / reshaping
Cache behavior
Attention implementation details
Framework overhead
```

The benchmark section of this repository will compare the measured tokens/sec against this theoretical bound and explain where the gap comes from.

---

# Sampling

The sampler supports:

## Temperature

Temperature changes the sharpness of the probability distribution.

```text
Lower temperature
→ more deterministic

Higher temperature
→ more diverse
```

The logits are transformed before sampling.

## Top-k

Keep only the `k` highest-probability candidate tokens.

```text
Vocabulary
    ↓
Top-k
    ↓
k candidates
    ↓
sample
```

## Top-p

Keep the smallest set of tokens whose cumulative probability exceeds `p`.

```text
Vocabulary
    ↓
sort by probability
    ↓
cumulative probability
    ↓
keep probability mass ≤ p
    ↓
sample
```

## Seeded RNG

Sampling uses an explicit seed so results can be reproduced.

```python
seed = 42
```

Running the same prompt with the same:

```text
model
prompt
sampling parameters
seed
```

should produce the same sampled output under the same runtime conditions.

---

# Benchmark Harness

All benchmark experiments are driven through:

```bash
python bench.py
```

The harness should:

1. Load the model.
2. Run warmup iterations.
3. Benchmark the requested context lengths.
4. Measure elapsed decode time.
5. Calculate tokens/sec.
6. Save raw results.
7. Generate plots.

The same command should regenerate the figures included in this README.

Example benchmark dimensions:

```text
Context lengths:
32
64
128
256
512
1024
```

The exact benchmark range can be adjusted depending on available hardware.

---

# Example Outputs

The repository will include examples showing:

### Greedy decoding

```text
Prompt:
"The capital of France is"

Greedy:
"The capital of France is Paris..."
```

### Sampled decoding

```text
Prompt:
"The capital of France is"

Temperature:
0.8

Top-k:
50

Top-p:
0.95

Seed:
42

Sampled:
"The capital of France is Paris..."
```

Additional prompts will be used to demonstrate cases where sampling produces more variation than greedy decoding.

---

# Project Structure

```text
.
├── qwen_from_scratch.py      # Custom Qwen implementation
├── bench.py                  # Reproducible benchmark harness
├── verify.py                 # Numerical correctness checks
├── generate.py               # Greedy + sampled generation
├── README.md
│
├── benchmarks/
│   ├── naive_context.png
│   ├── naive_vs_kv_cache.png
│   └── ...
│
└── results/
    ├── benchmark_results.json
    └── ...
```

The exact file organization may evolve as the implementation grows.

---

# Installation

```bash
pip install torch transformers matplotlib
```

Python 3.10+ recommended.

---

# Running

## Inspect the reference architecture

```bash
python inspect_model.py
```

## Verify the implementation

```bash
python verify.py
```

The main correctness requirement is:

```text
Maximum logit difference ≈ 1e-4 or better
```

## Run generation

```bash
python generate.py
```

## Run all benchmarks

```bash
python bench.py
```

`bench.py` is intended to regenerate the graphs and raw benchmark data used by the project.

---

# Development Progress

```text
[✓] Load Qwen2.5-0.5B
[✓] Build custom Transformer modules
[✓] Load pretrained weights
[✓] Verify logits against Hugging Face
[✓] Implement naive greedy decoding
[ ] Benchmark naive decoding vs context length
[ ] Implement KV cache
[ ] Benchmark naive vs KV cache
[ ] Analyze theoretical bandwidth ceiling
[ ] Implement temperature sampling
[ ] Implement top-k sampling
[ ] Implement top-p sampling
[ ] Add reproducible RNG
[ ] Add greedy vs sampled examples
[ ] Complete bench.py
[ ] Make all README graphs reproducible
```

---

# What This Project Is Meant to Demonstrate

This project is intentionally focused on the engineering details behind inference:

```text
Transformer mechanics
       +
Tensor shapes
       +
Pretrained weight loading
       +
Numerical verification
       +
Autoregressive decoding
       +
KV caching
       +
Sampling
       +
Performance measurement
       +
Hardware-aware analysis
```

Rather than treating inference as a black box, the project progressively exposes each component and measures the effect of optimizing it.

---

# Future Extensions

Possible follow-up experiments include:

```text
Flash Attention
SDPA vs eager attention
Quantization
FP16 / BF16 comparison
Batch decoding
Prefill vs decode performance
CPU vs GPU inference
Memory profiling
Different model sizes
```

---

# References

- Qwen2.5 model family
- Hugging Face Transformers
- PyTorch
- Original Qwen model implementation and documentation

This repository uses Hugging Face as a reference implementation and source of pretrained weights while implementing the inference path independently.

---

## License

Refer to the original Qwen2.5 model license and terms for the pretrained weights.

The custom implementation code in this repository is provided separately under the license chosen for this project.
