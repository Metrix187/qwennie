# qwennie v2 — CSS-native branch

v2 is not “v1 with bigger constants.” It changes the architecture around what a browser style engine can execute efficiently while preserving the fun constraint: **the deployed model runs entirely in CSS, with zero JavaScript.**

## Target architecture

- 64-token fixed causal sequence
- 3 decoder blocks
- d=80, SwiGLU MLP=160
- 5 query heads, 1 shared KV head (MQA), head dim 16
- RoPE + RMSNorm
- local causal attention window of 6 tokens
- two global assistant-boundary memory anchors
- two conversational turns in one CSS dependency graph
- int8 per-output-channel quantization
- 2:4 structured sparsity on CSS-heavy projection matrices
- hierarchical categorical softmax sampling using CSS `exp()`, `sign()`, `mod()`, and custom properties
- compositional prompt controls using native `<select>` elements and `:has(... option:checked)` selectors

The important number is not merely parameter count. MQA keeps the inherited KV cache to:

```text
3 layers × 63 cached positions × 16 KV dims × K/V = 6,048 properties
```

That is below v1's 6,720 KV-property footprint while increasing context length, layer count, width, and query-head count.

## Files

- `config.py` — shared architecture and training constants
- `extra_v2.txt` — two-turn/memory-focused training examples
- `train_v2.py` — deterministic PyTorch trainer + 2:4 projection + int8 exporter
- `verify_v2.py` — NumPy quantized reference inference and structural gates
- `build_v2.py` — exported weights → pure CSS runtime + HTML UI
- `site/` — generated output (created by the build; intentionally not required in source control)

The original root `corpus.txt` is reused as the single-turn personality corpus. `extra_v2.txt` is oversampled to teach the second-turn path and memory anchors.

## Build locally

Requires Python 3.11+, NumPy, and PyTorch.

```bash
python v2/train_v2.py
python v2/verify_v2.py
python v2/build_v2.py
python -m http.server -d v2/site 8472
```

Open `http://127.0.0.1:8472` in a recent Chromium build.

Training writes `v2/train_v2.pt` and `v2/weights_v2.json`. Verification writes `v2/expected_v2.json`. The compiler writes `v2/site/index.html`, `v2/site/model_v2.css`, and `v2/site/manifest.json`.

## Chunked training

The trainer saves the optimizer, model, RNG state, and current step. On constrained machines, train in chunks:

```bash
python v2/train_v2.py --max-run-steps 300
python v2/train_v2.py --resume --max-run-steps 300
# repeat until step 1800
```

The final chunk exports the CSS checkpoint automatically.

## Attention pattern

Most positions attend to the previous six tokens plus whichever assistant boundary anchors already exist. The first `<b>` anchor gets a wide read over turn-1 prompt tokens. The second `<b>` anchor gets a wide read covering the first anchor and turn-2 prompt region. This gives the generated second answer a compact causal route back to turn-1 state without paying quadratic full-history attention at every output position.

`weights_v2.json` stores the exact allowed-key list for every position. The compiler and NumPy reference consume that same list so the Python/CSS architecture cannot drift independently.

## Sampling

v1 used deterministic logit jitter followed by argmax. v2 does actual categorical sampling from `softmax(logits / temperature)`.

A single flat 400–500-way cumulative sum is unpleasant CSS, so v2 samples exactly in two stages:

1. sum token probabilities into fixed groups of up to 16;
2. sample a group from the group masses;
3. sample a token from that group's normalized probabilities.

This is mathematically equivalent to sampling from the original categorical distribution. Two deterministic LCG draws supply the uniform values, keeping builds/replays reproducible without JavaScript.

## Verification gates

`verify_v2.py` fails if a checkpoint loses the intended architecture, including:

- dense-equivalent parameter count below 190k;
- anything other than 1 KV head / at least 4 query heads;
- context below 64;
- local window above 8;
- inherited KV property budget >= 7,000;
- post-quantization sparse projection density above the tolerated 56% ceiling.

It then runs quantized NumPy reference conversations over multiple seeds and temperatures and writes `expected_v2.json`.

## Reproducibility

The training path seeds Python, NumPy, PyTorch, and the batch RNG. The CSS compiler performs no training or random mutation: a given `weights_v2.json` deterministically produces the same model structure and UI.

The GitHub Actions workflow in `.github/workflows/qwennie-v2.yml` can run the full train → verify → compile chain and upload the generated checkpoint/site as an artifact.

## Status

**Experimental v2 branch.** The architecture/compiler/training pipeline is intentionally isolated from v1 until browser parity and latency measurements are complete. v1 remains the known-good release on `main`.
