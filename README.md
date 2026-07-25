# qwennie 🐩

a 78,672-parameter **chat transformer that runs in pure css**. real decoder-only
architecture — 2 layers, 2 heads, d=48, rope, rmsnorm, swiglu, causal attention
with a kv cache — int8 matmul weights, trained from scratch to be a small
helpful puppygirl. you pick a message with a radio button and she writes her
reply word by word in a single style pass. there is not one line of javascript
on the page.

> **you:** who are you ?
> **qwennie:** qwennie ! seventy one thousand little numbers shaped like a dog .

(she has 78,672 parameters but believes she has seventy one thousand. she
counted twice. do not correct her.)

she is the little sister of [yipsy](../yipsy/README.md), the char-level mlp
next door. qwennie is the answer to "okay but could css do a *real* llm
architecture" — attention and all. yes. it can. it shouldn't, but it can.

## why not actual qwen or llama weights

people run llms in pdfs and font files, but those smuggle in a real js/wasm
engine. css gets no engine — every weight has to exist as a literal number
inside a calc() expression, and every attention step has to be unrolled as
rules. the smallest real qwen (0.5b) would be several gigabytes of stylesheet
before you even face the 150k-token vocabulary. the browser would simply pass
away. so qwennie keeps the architecture and shrinks the checkpoint until the
cascade can carry it: same shape as the big girls, pocket sized, trained on an
original corpus of 93 chat exchanges.

## try her

double-click `index.html` (chromium 125+, safari 18+, firefox 140+ — needs css
`mod()`, `sign()`, `exp()` and container style queries). a reply costs about
two seconds of style recalculation. she is thinking as hard as she can.

## what is actually happening

- **every token position is one level of dom nesting.** 10 prompt levels plus
  26 reply levels. the prompt radios write token ids onto the ten prefill
  wrappers; your message is tokenized at build time because css cannot read
  keyboards.
- **the transformer step is shared css.** one `.cl` rule block runs rmsnorm
  (`1 / sqrt(mean + eps)`), the q/k/v/mlp matmuls as int8 sums in calc(),
  swiglu (`x / (1 + exp(-x))` times the up projection), residuals, the lot.
  every nesting level re-evaluates it with its own inputs.
- **the kv cache is inherited custom properties.** each level writes its
  rotated key and value vectors into `--K{layer}p{pos}j{dim}` slots
  (registered, `inherits: true`), so every level below can attend to
  everything above. 6,720 registered properties of pure vibes.
- **attention is unrolled per position.** rope angles are compile-time
  literals per level, and level t's scores/softmax/weighted-sum are generated
  for exactly t+1 positions. o(t²) — in stylesheet bytes. that's why
  model.css is 8 mb. softmax runs on css `exp()` with max-subtraction, real
  division, no tricks.
- **decoding is the yipsy trick at transformer scale:** logits + a per-glyph
  jitter from an lcg (`mod(137s + 29, 251)`, also in calc) + argmax via
  `sign()`. temperature radios scale the jitter. the `<e>` end token sets a
  done-flag that travels down the nesting and blanks every glyph after it,
  so she actually stops talking.
- **int8:** all matmul weights live in weights.json as int8 with per-channel
  scales and are dequantized into the calc() terms at build time. norm gains
  and embeddings stay small-float, gguf style.

## she is verified against numpy

`build.py` writes `expected.json`: the reference output of all 144
message × reroll × temperature combos. serve the folder
(`python -m http.server -d . 8472 --bind 127.0.0.1`), open the page, paste this
in the console, wait ~6 minutes:

```js
(async () => {
  const exp = await (await fetch('/expected.json')).json();
  const gs = [...document.querySelectorAll('.g')];
  const read = () => gs.map(g => {
    const c = getComputedStyle(g, '::before').content;
    return c.startsWith('"') ? c.slice(1, -1) : '@';
  }).join('');
  let pass = 0, fails = [];
  for (const [k, want] of Object.entries(exp.combos)) {
    k.split('|').forEach(id => document.getElementById(id).checked = true);
    await new Promise(r => setTimeout(r));
    read() === want ? pass++ : fails.push(k);
  }
  console.log(pass + '/144 pass', fails);
})();
```

144/144 when i shipped this: every reply the stylesheet renders is
byte-identical to the python forward pass, through two layers of attention,
softmax, rmsnorm and swiglu. the cascade is really running a transformer.

## retrain her

```
python train.py   # gradchecked numpy transformer, ~15 min on cpu, writes weights.json
python build.py   # bakes weights.json into model.css + index.html + expected.json
```

the corpus is `corpus.txt`, one `question || reply` per line, and the twelve
ui messages live at the top of `build.py` — they have to be questions the
corpus taught her or she will answer in confident nonsense (honestly also
cute). loss around 0.01 means she has memorized her personality, which at
78k parameters is the entire point.

## files

| file | what |
| --- | --- |
| `index.html` | generated. chat ui + 36 levels of nested spans |
| `model.css` | generated. her entire brain, ~8 mb (attention is o(t²) in bytes) |
| `style.css` | hand-written theme, safe to edit |
| `corpus.txt` | original training chats, one exchange per line |
| `train.py` | numpy transformer with hand-derived, gradchecked backprop |
| `build.py` | weights → stylesheet compiler |
| `weights.json` | int8 weights + scales + rope tables |
| `expected.json` | generated. ground truth for the parity check |

## license

wtfpup. do what the fuck you want to, pup. the name is a homage to a much
bigger dog; no actual qwen weights, code, or tokenizers were used, involved,
or harmed.
