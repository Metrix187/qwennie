# qwennie v2 — CSS-native transformer experiment

v2 keeps the rule that makes qwennie interesting: **deployed inference happens entirely in the browser CSS style engine, with zero JavaScript.** The training and verification tools are ordinary Python; the generated page is not.

## Final architecture

- **205,040 dense-equivalent parameters**
- **3 decoder layers**, `d=80`, SwiGLU `d_ff=160`
- **5 query heads / 1 shared KV head** (MQA), head dim 16
- **520 word-level tokens**
- **64 causal positions**
- **two user turns** in one static causal dependency tree
- **6-token local causal attention + two `<b>` memory anchors**
- RoPE + RMSNorm + SwiGLU
- int8 per-output-channel quantization
- 2:4 structured sparsity on CSS-heavy transformer projections
- exact hierarchical categorical sampling from `softmax(logits / temperature)`
- native `<select>` compositional prompts; the exported UI exposes the 128 most-used prompt tokens

The final sparse/int8 export contains **123,376 nonzero quantized weights out of 204,480 matrix weights**. The sparsity-eligible transformer projections contain **80,584 / 161,280 nonzero weights** after quantization.

The generated runtime is currently **15,476,395 bytes raw** and about **1.57 MB gzip**.

## Why v2 does not just make v1 wider

The v1 bottleneck was the browser dependency graph, not merely parameter count. v2 changes the architecture around CSS.

### Multi-query attention

Only one K/V head is cached and shared by five query heads:

```text
3 layers × 63 cached positions × 16 KV dims × K/V
= 6,048 inherited KV custom properties
```

That is *less* KV state than v1's 6,720 properties even though v2 has 64 positions, three layers, greater width, and more query heads.

### Local attention + memory anchors

Ordinary positions attend to the previous six causal positions plus completed assistant-boundary anchors. The first `<b>` gets a wide read over turn-one prompt state. The second `<b>` gets a deliberate wide read over turn one + the second prompt. Turn-two generation can therefore retrieve conversation state without paying full-history attention at every generated position.

The exact allowed-key list is exported into `weights_v2.json`; both the NumPy reference and CSS compiler consume it.

## CSS-native token embeddings

An early v2 prototype used `@container style(--tk: …)` as token-id → embedding lookup. Chromium exposed a nasty engine limit: after a short chain of autoregressively dependent style queries, later embeddings could stop propagating correctly.

v2 no longer uses style queries in the autoregressive token lookup path.

Each token ID is transformed by an invertible affine map modulo 1024, represented as a unique 10-bit code, and embedded as the learned sum of ten per-bit state vectors:

```text
code = (token_id * 405 + 17) mod 1024
embedding(token) = scale * Σ bit_table[position, bit_value]
```

The model is trained with this representation directly. It is not a post-training approximation. In CSS, extracting ten bits and summing learned bit deltas is ordinary custom-property arithmetic, so autoregressive generation no longer depends on nested token style queries.

Visible integer-token → word conversion also avoids style queries: the compiler emits a native CSS `@counter-style` whose symbols are the vocabulary.

## Real sampling

v1 used deterministic logit jitter + argmax. v2 computes real temperature-scaled softmax probabilities.

A flat 520-token inverse CDF would make another long browser dependency chain, so sampling is exactly factorized into two stages:

1. compute exponentiated token masses;
2. sum them into groups of at most 16 tokens;
3. sample a group using one deterministic LCG uniform;
4. sample within that group using the next uniform.

This produces the same categorical distribution as a flat softmax sampler, subject to browser floating-point behavior.

## Training

Requires Python, NumPy and PyTorch.

```bash
python v2/train_v2.py --steps 1800
```

Training is deterministic on CPU. The first 60% is dense; during the last 40%, eligible linear layers are projected back onto 2:4 sparsity after every optimizer step so the model learns around the exact zeros that make the stylesheet cheaper.

For environments with short command limits:

```bash
python v2/train_v2.py --steps 1800 --max-run-steps 300
python v2/train_v2.py --steps 1800 --resume --max-run-steps 300
# repeat until step 1800
```

The root `corpus.txt` supplies v1 personality/single-turn examples. `extra_v2.txt` adds and oversamples two-turn/memory behavior.

## Verify and build

```bash
python v2/verify_v2.py
python v2/build_v2.py
python -m http.server -d v2/site 8472
```

The reference verifier exercises **36 quantized two-turn conversations** over two seeds, three temperatures and six prompt pairs, while enforcing the architectural gates (MQA, context, vocabulary, parameter count and KV budget).

The final 205,040-parameter checkpoint passed all 36 NumPy/int8 reference cases. Headless Chromium was also checked token-for-token against those reference IDs on representative cases covering all six prompt pairs at seed 17 / temperature 0.45, plus a second-seed high-temperature path. Those checks include the cross-turn `remember snow` → `what did i say ?` memory case and the compositional `can we chat twice ?` prompt. The large stylesheet makes an exhaustive 36-case browser sweep expensive in a constrained headless sandbox, so browser verification is kept as a separate gate rather than being disguised as a cheap unit test.

## Generated files

`build_v2.py` writes:

- `v2/site/index.html` — two-turn no-JS UI and static causal DOM
- `v2/site/model_v2.css` — the complete inference runtime + checkpoint
- `v2/site/manifest.json` — footprint and model metadata

`train_v2.py` writes `weights_v2.json`; `verify_v2.py` writes `expected_v2.json`.

Generated checkpoints/site files are intentionally ignored on this branch. `.github/workflows/qwennie-v2.yml` can run train → verify → compile and upload the result as a workflow artifact.

## Status

**Experimental, browser-running v2.** The final bit-coded runtime fixes the recursive style-query failure found during Chromium testing, the NumPy/int8 reference is 36/36 green, and representative Chromium renders are token-for-token identical to the reference. v1 on `main` remains the known-good published release until v2 latency/parity testing is broad enough to promote.
