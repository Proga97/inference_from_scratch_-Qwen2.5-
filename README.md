# Qwen2.5-0.5B Inference From Scratch

A from-scratch PyTorch implementation of **Qwen2.5-0.5B inference**, built to understand how Transformer inference works internally rather than relying on Hugging Face generation utilities.

The project loads the pretrained Qwen2.5-0.5B weights into a custom implementation and verifies that its logits closely match the official Hugging Face model.

## Goals

This project focuses on two core inference tasks:

1. **Reimplement Qwen2.5-0.5B forward inference**
   - Build the Transformer architecture manually.
   - Load the official pretrained weights into the custom modules.
   - Verify that the custom implementation reproduces Hugging Face logits to approximately `1e-4`.

2. **Implement naive greedy decoding**
   - Generate tokens one at a time without `model.generate()`.
   - Run a full-sequence forward pass for every generated token.
   - Benchmark tokens/sec as context length increases.
   - Demonstrate the performance cost of naive autoregressive decoding.

---

## Architecture

The implementation follows the Qwen2.5-0.5B decoder architecture:

```text
Input IDs
   │
   ▼
Token Embeddings
   │
   ▼
┌───────────────────────────────────┐
│ Transformer Decoder Layer × N     │
│                                   │
│   RMSNorm                         │
│      │                            │
│      ▼                            │
│   Self-Attention                  │
│      │                            │
│      ├── Q Projection             │
│      ├── K Projection             │
│      ├── V Projection             │
│      ├── RoPE                     │
│      ├── Grouped Query Attention  │
│      ├── Attention Scores         │
│      ├── Causal Mask              │
│      ├── Softmax                  │
│      └── Output Projection        │
│                                   │
│   Residual Connection             │
│                                   │
│   RMSNorm                         │
│      │                            │
│      ▼                            │
│   SwiGLU MLP                      │
│      ├── gate_proj                │
│      ├── SiLU                     │
│      ├── up_proj                  │
│      ├── Element-wise Multiply    │
│      └── down_proj               │
│                                   │
│   Residual Connection             │
└───────────────────────────────────┘
   │
   ▼
Final RMSNorm
   │
   ▼
LM Head
   │
   ▼
Logits
```

---

## Components Implemented

### Token Embeddings

Converts token IDs into learned hidden representations.

```text
[batch, sequence]
        ↓
[batch, sequence, hidden_size]
```

### RMSNorm

Implements root mean square normalization with learned scaling weights.

\[
\text{RMSNorm}(x)
=
\frac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}
\times \text{weight}
\]

### Self-Attention

The attention implementation includes:

- Query / Key / Value projections
- Multi-head reshaping
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- Scaled dot-product attention
- Causal masking
- Softmax
- Attention-weighted Value aggregation
- Output projection

### Rotary Position Embeddings

RoPE is applied to Query and Key representations to encode token positions.

### Grouped Query Attention

Qwen2.5-0.5B uses fewer Key/Value heads than Query heads.

The implementation expands the KV heads to match the Query heads before attention score computation.

### SwiGLU MLP

The feed-forward network follows the gated structure:

\[
\text{MLP}(x)
=
\text{down\_proj}
\left(
\text{SiLU}(\text{gate\_proj}(x))
\odot
\text{up\_proj}(x)
\right)
\]

### Residual Connections

Each decoder block follows the pre-normalization structure:

```text
x
↓
RMSNorm
↓
Attention
↓
Residual Add
↓
RMSNorm
↓
MLP
↓
Residual Add
```

---

## Weight Loading

Instead of training the model, the project reuses the pretrained Qwen2.5-0.5B parameters.

The custom architecture mirrors the Hugging Face module hierarchy so the checkpoint can be loaded directly with:

```python
my_model.load_state_dict(
    hf_model.state_dict(),
    strict=True
)
```

This ensures the custom implementation uses the same learned parameters as the original model.

---

## Verification

Each major component was verified against the Hugging Face implementation before assembling the full model.

### Verified Components

```text
Embedding                  ✅
RMSNorm                    ✅
Q Projection               ✅
K Projection               ✅
V Projection               ✅
Attention Head Reshaping   ✅
RoPE                       ✅
GQA / repeat_kv            ✅
Attention                  ✅
SwiGLU MLP                 ✅
Decoder Layer              ✅
Full Model                 ✅
Final Logits               ✅
```

### Logit Verification

The final validation compares the complete logits produced by:

```text
Official Hugging Face Qwen2.5-0.5B
                  vs
Custom Qwen2.5-0.5B implementation
```

The target is:

```text
max(abs(HF logits - custom logits)) ≈ 1e-4
```

This verifies that the custom forward pass reproduces the original model numerically.

---

## Naive Greedy Decoding

The project intentionally implements the simplest possible autoregressive decoding strategy.

For every generated token:

```text
Current sequence
      │
      ▼
Full Transformer forward pass
      │
      ▼
Last-token logits
      │
      ▼
argmax
      │
      ▼
Next token
      │
      ▼
Append token
      │
      └───────────────► repeat
```

No:

```python
model.generate()
```

No Hugging Face pipeline.

No KV cache.

This means every new token recomputes the entire previous context.

---

## Why Naive Decoding Gets Slow

Self-attention operates over the full sequence.

For a context of length `N`, attention involves roughly:

\[
O(N^2)
\]

pairwise interactions.

With naive decoding, this full computation is repeated for every generated token.

Conceptually:

```text
Step 1 → forward(context = N)
Step 2 → forward(context = N + 1)
Step 3 → forward(context = N + 2)
...
```

As the context grows, the number of computations increases rapidly.

This is why tokens/sec decreases as context length increases.

---

## Benchmark

The benchmark measures:

```text
Context Length → Tokens / Second
```

Example experiment:

```text
32
64
128
256
512
1024
```

The benchmark keeps the number of newly generated tokens fixed and measures the time required to generate them.

The resulting plot illustrates the performance degradation of naive full-sequence decoding.

---

## Project Structure

```text
.
├── my_model.py
├── inspect_model.py
├── verify.py
├── benchmark.py
├── naive_generation_benchmark.png
└── README.md
```

Your exact filenames may differ depending on how you organize the implementation.

---

## Requirements

Python 3.10+

Install dependencies:

```bash
pip install torch transformers matplotlib
```

---

## Running the Project

### Inspect the Qwen architecture

```bash
python inspect_model.py
```

### Run the custom model

```bash
python my_model.py
```

### Verify logits

Run the verification code to compare your implementation against Hugging Face.

The key validation is:

```text
Maximum difference < 1e-4
```

### Run naive generation

Generate text using the custom model:

```text
Prompt
  ↓
Custom Qwen forward()
  ↓
Greedy argmax
  ↓
Append token
  ↓
Repeat
```

### Run the benchmark

Measure generation speed at different context lengths:

```bash
python benchmark.py
```

The benchmark produces a context-length vs tokens/sec plot.

---

## Design Philosophy

The goal of this project is **not** to create a faster implementation than Hugging Face.

The goal is to understand what is happening underneath the abstractions.

Instead of:

```python
model.generate(...)
```

the project explicitly implements:

```text
Embedding
→ Transformer Layers
→ Attention
→ RoPE
→ GQA
→ MLP
→ Residual Connections
→ LM Head
→ Greedy Decoding
```

Every major component is independently verified against the reference model before being composed into the full system.

---

## What This Demonstrates

This project demonstrates hands-on understanding of:

- Transformer architecture
- PyTorch tensor shapes and broadcasting
- Attention mechanics
- Multi-head attention
- Grouped Query Attention
- Rotary Position Embeddings
- RMSNorm
- SwiGLU
- Residual connections
- Pretrained weight loading
- Numerical verification
- Autoregressive decoding
- Attention complexity
- Inference benchmarking

---

## Future Work

Potential extensions include:

- KV caching
- Incremental decoding
- SDPA-based attention
- Flash Attention
- Prefill vs decode benchmarking
- Memory profiling
- Batch generation
- Temperature / top-k / top-p sampling
- CPU vs GPU benchmarking
- Quantized inference

The most natural next optimization is **KV caching**, which avoids recomputing Key and Value representations for the entire context at every generation step.

---

## Acknowledgements

This project uses the pretrained:

**Qwen2.5-0.5B** model from Alibaba Cloud's Qwen team via Hugging Face Transformers.

The Hugging Face implementation is used as a reference for architecture inspection, pretrained weights, and numerical verification.

---

## License

This repository contains custom implementation code built around the publicly available Qwen2.5-0.5B pretrained model.

Refer to the original Qwen model repository and license terms for the model weights and associated usage conditions.