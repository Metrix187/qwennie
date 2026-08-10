# qwennie

![Status: Complete](https://img.shields.io/badge/status-complete-blue)
![License: WTFPUP](https://img.shields.io/badge/license-WTFPUP-pink)
![No JavaScript](https://img.shields.io/badge/javascript-none-lightgrey)

A 78,672-parameter decoder-only transformer that runs entirely in CSS. Real architecture — 2 layers, 2 heads, d=48, RoPE, RMSNorm, SwiGLU, causal attention with a KV cache — int8 quantized, trained from scratch. Pick a message with a radio button; she writes her reply word-by-word in a single style pass. Zero JavaScript.

---

## Why this exists

People run LLMs in PDFs and font files, but those smuggle in a real JS/Wasm engine. CSS gets no engine — every weight has to be a literal number inside `calc()`, and every attention step has to be unrolled as rules. This project answers "okay but could CSS do a *real* LLM architecture — attention and all?" The answer is yes. It shouldn't. It can.

The smallest real Qwen (0.5B) would be several gigabytes of stylesheet before you face the 150k-token vocabulary. qwennie keeps the architecture and shrinks the checkpoint until the cascade can carry it: same shape, pocket-sized, trained on an original corpus of 93 chat exchanges.

## Try it

Double-click `index.html`. Requires Chromium 125+, Safari 18+, or Firefox 140+ (needs CSS `mod()`, `sign()`, `exp()`, and container style queries). A reply costs ~2 seconds of style recalculation. No server. No build step.

## What is actually happening

- **Every token position is one level of DOM nesting.** 10 prompt + 26 reply levels. Prompt radios write token IDs onto the prefill wrappers; messages are tokenized at build time because CSS cannot read keyboards.
- **The transformer step is shared CSS.** One `.cl` rule block runs RMSNorm, Q/K/V/MLP matmuls as int8 sums in `calc()`, SwiGLU, and residuals — re-evaluated at every nesting level with its own inputs.
- **The KV cache is inherited custom properties.** Each level writes rotated key and value vectors into `--K{layer}p{pos}j{dim}` slots (`inherits: true`). 6,720 registered properties.
- **Attention is unrolled per position.** RoPE angles are compile-time literals; softmax uses CSS `exp()` with max-subtraction. O(t²) in stylesheet bytes — that's why `model.css` is 8 MB.
- **Decoding:** logits + per-glyph LCG jitter (`mod(137s+29, 251)`) + argmax via `sign()`. Temperature radios scale the jitter. The `<e>` end token sets a done-flag that blanks every glyph after it.
- **int8:** matmul weights are int8 with per-channel scales, dequantized into `calc()` at build time.

## Verified against NumPy

`build.py` writes `expected.json` — reference outputs for all 144 message × reroll × temperature combos. 144/144 pass: every reply the stylesheet renders is byte-identical to the Python forward pass through two layers of attention, softmax, RMSNorm, and SwiGLU.

## Retrain

```bash
python train.py  # NumPy transformer, gradchecked backprop, ~15 min CPU → writes weights.json
python build.py  # bakes weights.json into model.css + index.html + expected.json
```

Edit `corpus.txt` (one `question || reply` per line) to change her personality.

## Files

| File | What |
|------|------|
| `index.html` | Generated — chat UI + 36 levels of nested spans |
| `model.css` | Generated — her entire brain (~8 MB) |
| `style.css` | Hand-written theme |
| `corpus.txt` | Original training chats |
| `train.py` | NumPy transformer with hand-derived backprop |
| `build.py` | Weights → stylesheet compiler |
| `weights.json` | int8 weights + scales + RoPE tables |
| `expected.json` | Ground truth for parity check |

## Status

Complete. Model trained and verified. Architecture fixed; corpus can be extended and model retrained.

## License

[WTFPUP](LICENSE) — do what the fuck you want to, pup. No actual Qwen weights, code, or tokenizers were used.
