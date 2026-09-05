# qwennie

![Status: Complete](https://img.shields.io/badge/status-complete-blue)
![License: WTFPUP](https://img.shields.io/badge/license-WTFPUP-pink)
![No JavaScript](https://img.shields.io/badge/javascript-none-lightgrey)

A 78,672-parameter decoder-only transformer that runs entirely in CSS. Real architecture — 2 layers, 2 heads, d=48, RoPE, RMSNorm, SwiGLU, causal attention with a KV cache — int8 quantized, trained from scratch. Pick a message with a radio button; she writes her reply word-by-word in a single style pass. Zero JavaScript.

**[Try her live →](https://metrix187.github.io/qwennie/)** · **[Read the full writeup →](https://quantara.cv/articles/qwennie-css-transformer.html)** · **[Meet v2 →](https://metrix187.github.io/qwennie/v2/site/)**

> **you:** who are you ?
> **qwennie:** qwennie! seventy one thousand little numbers shaped like a dog.

(She has 78,672 parameters but believes she has seventy one thousand. She counted twice. Do not correct her.)

---

## Why this exists

People run LLMs in PDFs and font files, but those smuggle in a real JS/Wasm engine. CSS gets no engine — every weight has to be a literal number inside `calc()`, and every attention step has to be unrolled as rules. This project answers "okay but could CSS do a *real* LLM architecture — attention and all?" The answer is yes. It shouldn't. It can.

The smallest real Qwen (0.5B) would be several gigabytes of stylesheet before you face the 150k-token vocabulary. qwennie keeps the architecture and shrinks the checkpoint until the cascade can carry it: same shape, pocket-sized, trained on an original corpus of 93 chat exchanges.

Her big sister [yipsy](https://github.com/Metrix187/yipsy) is a char-level MLP in the same substrate.

## Try it

Open the [live demo](https://metrix187.github.io/qwennie/), or clone and double-click `index.html`. **Use a Chromium browser (138+).** She needs CSS `mod()`, `sign()`, `abs()`, `exp()`, `sqrt()`, registered custom properties (`@property`), and container style queries on custom properties. Feature-wise that's Chromium 138+, Safari 18+, Firefox 151+ — `sign()`/`abs()` gate Chromium, style queries gate the other two.

But feature support isn't the whole story, and this is the honest bit: **she only works properly in Chromium.** The first render is correct in all three engines — WebKit and Gecko compute the same reply Chromium does, byte for byte. What they don't do is *re-compute* when you pick a different message. In WebKit the reply doesn't change at all; in Gecko it changes halfway, so you get the front of the old reply welded to the end of the new one. Invalidating a custom-property chain 36 levels deep is apparently a place the engines disagree.

Tested with Playwright's WebKit 26.5 and Firefox 153 rather than shipping Safari/Firefox, so treat it as strong evidence and not proof — but don't expect her to behave outside Chromium.

A reply costs about 2.5 seconds of style recalculation (measured p50 on Chrome 148, desktop). No server. No build step.

## What is actually happening

- **Every token position is one level of DOM nesting.** 10 prompt + 26 reply levels. Prompt radios write token IDs onto the prefill wrappers; messages are tokenized at build time because CSS cannot read keyboards.
- **The transformer step is shared CSS.** One `.cl` rule block runs RMSNorm, Q/K/V/MLP matmuls as int8 sums in `calc()`, SwiGLU, and residuals — re-evaluated at every nesting level with its own inputs.
- **The KV cache is inherited custom properties.** Each level writes rotated key and value vectors into `--K{layer}p{pos}j{dim}` slots (`inherits: true`). 6,720 registered properties for the cache alone; 9,187 in the stylesheet overall.
- **Attention is unrolled per position.** RoPE angles are compile-time literals; softmax uses CSS `exp()` with max-subtraction. O(t²) in stylesheet bytes — that's why `model.css` is 8 MB, though it gzips to about 900 KB, which is what you actually download.
- **Decoding:** logits + per-glyph LCG jitter (`mod(137s+29, 251)`) + argmax via `sign()`. Temperature radios scale the jitter. The `<e>` end token sets a done-flag that blanks every glyph after it.
- **int8:** matmul weights are int8 with per-channel scales, dequantized into `calc()` at build time.

## Verified against NumPy

`build.py` writes `expected.json` — reference outputs for all 144 message × reroll × temperature combos, straight from the Python forward pass. To check the stylesheet against it yourself, serve the folder and run the harness in the console:

```bash
python -m http.server -d . 8472 --bind 127.0.0.1
```

```js
(async () => {
  const gs = [...document.querySelectorAll('.g')];
  const read = () => gs.map(g => {
    const c = getComputedStyle(g, '::before').content;
    return c.startsWith('"') ? c.slice(1, -1) : '@';
  }).join('');
  const exp = (await (await fetch('/expected.json')).json()).combos;
  let pass = 0; const fails = [];
  for (const [k, want] of Object.entries(exp)) {
    document.querySelectorAll('input[type=radio]').forEach(x => x.checked = false);
    read();
    k.split('|').forEach(id => document.getElementById(id).checked = true);
    read() === want ? pass++ : fails.push(k);
    await new Promise(r => setTimeout(r));
  }
  console.log(`${pass}/${Object.keys(exp).length} pass`, fails);
})();
```

144/144, in about ten minutes. Every reply the stylesheet renders is byte-identical to the Python forward pass through two layers of attention, softmax, RMSNorm and SwiGLU.

**It is not a lookup table, and you can check that in one command:**

```bash
grep -c "seventy one" model.css   # 0 — no reply phrase exists in the stylesheet
grep -c "seventy"     model.css   # 1 — each vocab word appears exactly once, as a glyph rule
```

The sentences only exist in `expected.json`, which is the answer key, not an input.

## Retrain

```bash
python train.py  # NumPy transformer, gradchecked backprop, ~30 min CPU → writes weights.json
python build.py  # bakes weights.json into model.css + index.html + expected.json
```

**The whole pipeline is deterministic, corpus to stylesheet.** `train.py` is seeded, so
retraining from `corpus.txt` reproduces `weights.json` byte-for-byte — verified on a
different NumPy version from the one she was trained on — and `build.py` on those weights
reproduces `model.css` byte-for-byte (sha256 `44b10e1a…`). Nothing in the stylesheet was
placed by hand, and you can rebuild her from the corpus up and get literally the same dog.
`train.py` also gradient-checks its hand-derived backprop against finite differences at
startup before it will train (worst relative error 2.1e-05).

Edit `corpus.txt` (one `question || reply` per line) to change her personality. The twelve UI messages at the top of `build.py` have to be questions the corpus taught her, or she answers in confident nonsense (honestly also cute).

## v2 — two turns, 205k parameters

`v2/` is the second pass: **205,040 parameters, a 64-token context, and a two-turn conversation** where you build the prompt word-by-word from dropdowns instead of picking one of twelve canned messages. Still zero JavaScript.

**[Try v2 live →](https://metrix187.github.io/qwennie/v2/site/)**

> **you:** who are you ?
> **qwennie:** i am qwennie. a tiny transformer living in css.
> **you:** how do you work ?
> **qwennie:** little sums flow through the cascade. the browser does my thinking.

| | v1 | v2 |
|---|---|---|
| Parameters | 78,672 | 205,040 |
| Layers / d / MLP | 2 / 48 / 96 | 3 / 80 / 160 |
| Context | 26 | 64 |
| Attention | 2 heads, dense causal | 5 query / 1 KV head, sliding window + anchors |
| Sampling | argmax + LCG jitter | exact categorical, two-stage inverse CDF |
| KV cache | 6,720 properties | 6,048 properties |
| Stylesheet | ~8 MB | 15.5 MB (1.5 MB gzipped) |

The KV cache got *smaller* while the model got almost three times bigger. That is the whole trick:

- **Grouped-query attention** — 5 query heads share 1 KV head, head dim 16.
- **Sliding window + memory anchors.** Each position attends to the last 6 tokens plus the two `<b>` turn markers, which act as conversation summaries. 465 attention terms across the whole sequence instead of the 2,080 a dense causal mask needs. v1's O(t²) is why `model.css` was already 8 MB at 26 positions — without this, 64 positions would not fit at all.
- **10-bit binary token embeddings.** Token ids are scrambled through an invertible affine map mod 1024 and spelled out as ten bits; the embedding is the scaled sum of one learned vector per bit state. 520 tokens out of 20 vectors.
- **2:4 sparsity** on the transformer projections, int8 with per-output-channel scales. 123,376 of 204,480 quantized weights are nonzero; the zeros never reach the `calc()` sums.
- **Real categorical sampling.** Logits are softmaxed, bucketed into 33 groups of 16, then picked by a two-stage inverse-CDF lookup with two LCG draws. A genuine multinomial sample, in CSS.
- **Two turns.** Turn 2 resets the done-flag and derives a fresh deterministic substream from the root seed (`mod(rootseed*193 + 17, 251)`), so she answers a follow-up while still remembering turn one.

A reply costs about twelve seconds of style recalculation. She is thinking as hard as she can.

### v2 verified against NumPy

`v2/verify_v2.py` is the int8 reference forward pass in NumPy. It writes `v2/expected_v2.json` — every token id of all 64 positions, for 6 message pairs × 2 seeds × 3 temperatures. `v2/browser_parity.py` drives headless Chromium and diffs what the stylesheet actually computed against it.

```bash
python v2/train_v2.py   # writes weights_v2.json (generated, not committed)
python v2/verify_v2.py  # writes expected_v2.json
python -m playwright install chromium
python v2/browser_parity.py --count 0   # 0 = every case; it batches 6 at a time by default
```

36/36 at ship time. Every token id across both turns is identical to the Python forward pass — through grouped-query attention, a sparse causal mask, softmax, RMSNorm, SwiGLU and the categorical sampler. `build_v2.py` also reproduces the shipped `model_v2.css` byte-for-byte from `weights_v2.json`.

The suite covers the `ball` and `bone` rerolls. `duck` and `zoom` take the same code path with different seed constants — TODO: fold them into `expected_v2.json` so all four are covered.

### Retrain v2

```bash
python v2/train_v2.py  # PyTorch, seeded → weights_v2.json + a resumable train_v2.pt
python v2/build_v2.py  # bakes it into v2/site/
```

Training data is `corpus.txt` plus `v2/extra_v2.txt`, which holds the two-turn exchanges as `u1 || a1 || u2 || a2` per line.

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

v2 lives in `v2/` with the same shape — `config.py` holds the architecture constants, `train_v2.py` / `build_v2.py` / `verify_v2.py` are the pipeline, `browser_parity.py` is the Chromium diff, and the generated page is in `v2/site/`. Her weights, ground truth and training checkpoint are generated artifacts and are not committed; `train_v2.py` is seeded, so they rebuild deterministically.

## Status

Complete. Both models trained and verified against the NumPy reference. Architecture fixed; the corpus can be extended and either model retrained.

## License

[WTFPUP](LICENSE) — do what the fuck you want to, pup. No actual Qwen weights, code, or tokenizers were used.
