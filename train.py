# qwennie's brain gym. trains a from scratch micro llama on corpus.txt and
# dumps int8 weights to weights.json for build.py.
#
# the real deal, just doll sized: decoder only transformer, rope, rmsnorm,
# swiglu, causal attention. everything backward is derived by hand and grad
# checked against finite differences at startup, because numpy does not
# forgive and neither does a stylesheet.

import json
import math
import numpy as np

# fixed sequence layout, shared with build.py:
# positions 0..9 are the prompt: [<p> pads] <u> question words <b>
# positions 10..35 are the reply: words . <e> <e> <e> ...
T = 36
PROMPT = 10
GEN = T - PROMPT

D = 48
L = 2
HEADS = 2
HD = D // HEADS
MLP = 96
ROPE_BASE = 10000.0
EPS = 1e-5

STEPS = 6000
BATCH = 32
LR = 3e-3
WD = 0.01

rng = np.random.default_rng(11)


# --- data -------------------------------------------------------------------

def load_pairs(path="corpus.txt"):
    pairs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or "||" not in line:
            continue
        q, a = [s.strip() for s in line.split("||")]
        pairs.append((q.split(), a.split()))
    return pairs


pairs = load_pairs()
words = sorted({w for q, a in pairs for w in q + a})
vocab = ["<p>", "<u>", "<b>", "<e>"] + words
V = len(vocab)
stoi = {w: i for i, w in enumerate(vocab)}
PAD, USR, BOT, END = 0, 1, 2, 3
print(f"{len(pairs)} pairs, vocab {V}")
assert V <= 384, "vocab too fat for the stylesheet"  # 337 as written. she is wordy

for q, a in pairs:
    assert len(q) <= PROMPT - 2, f"question too long: {q}"
    assert len(a) <= GEN - 2, f"reply too long: {a}"


def encode(q, a=None):
    ids = [PAD] * (PROMPT - 2 - len(q)) + [USR] + [stoi[w] for w in q] + [BOT]
    if a is not None:
        ids += [stoi[w] for w in a] + [END] * (T - len(ids) - len(a))
    return ids


X = np.array([encode(q, a) for q, a in pairs])
# loss only on reply predictions: the words, the first <e>, and one extra <e>
# so the end state is absorbing. the rest of the padding is dead weight.
M = np.zeros((len(pairs), T - 1))
for r, (q, a) in enumerate(pairs):
    M[r, PROMPT - 1 : PROMPT + len(a) + 1] = 1.0


# --- params -----------------------------------------------------------------

def init(v=V, d=D, mlp=MLP):
    p = {"emb": rng.normal(0, 0.02, (v, d)), "gf": np.ones(d), "lm": rng.normal(0, 0.02, (d, v))}
    for l in range(L):
        p[f"g1{l}"] = np.ones(d)
        p[f"g2{l}"] = np.ones(d)
        for n in ("wq", "wk", "wv", "wo"):
            p[f"{n}{l}"] = rng.normal(0, 0.02, (d, d))
        p[f"wg{l}"] = rng.normal(0, 0.02, (d, mlp))
        p[f"wu{l}"] = rng.normal(0, 0.02, (d, mlp))
        p[f"wd{l}"] = rng.normal(0, 0.02, (mlp, d))
    return p


def rope_tables(t, hd, base=ROPE_BASE):
    inv = base ** (-np.arange(hd // 2) / (hd // 2))
    ang = np.arange(t)[:, None] * inv[None, :]
    return np.cos(ang), np.sin(ang)  # (T, HD/2)


COS, SIN = rope_tables(T, HD)


def rope(x, cos, sin):
    # x: (B,T,H,HD), rotate (even, odd) pairs
    xe, xo = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    out = np.empty_like(x)
    out[..., 0::2] = xe * c - xo * s
    out[..., 1::2] = xe * s + xo * c
    return out


def rope_back(d, cos, sin):
    de, do = d[..., 0::2], d[..., 1::2]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    out = np.empty_like(d)
    out[..., 0::2] = de * c + do * s
    out[..., 1::2] = -de * s + do * c
    return out


def rmsnorm_f(x, g):
    r = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
    return x * r * g, r


def rmsnorm_b(dy, x, g, r):
    dg = (dy * x * r).sum((0, 1))
    dxh = dy * g
    d = x.shape[-1]
    dx = r * dxh - x * (r ** 3) * ((dxh * x).sum(-1, keepdims=True) / d)
    return dx, dg


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward(p, idx, cos, sin, want_cache=False):
    B, t = idx.shape
    x = p["emb"][idx]
    cache = {"idx": idx, "x0": x}
    for l in range(L):
        c = {}
        c["xin"] = x
        a, c["r1"] = rmsnorm_f(x, p[f"g1{l}"])
        c["a"] = a
        q = (a @ p[f"wq{l}"]).reshape(B, t, HEADS, HD)
        k = (a @ p[f"wk{l}"]).reshape(B, t, HEADS, HD)
        v = (a @ p[f"wv{l}"]).reshape(B, t, HEADS, HD)
        c["q"], c["k"], c["v"] = q, k, v
        qr, kr = rope(q, cos, sin), rope(k, cos, sin)
        c["qr"], c["kr"] = qr, kr
        s = np.einsum("bthd,bshd->bhts", qr, kr) / math.sqrt(HD)
        s = np.where(np.tril(np.ones((t, t), bool))[None, None], s, -1e9)
        s -= s.max(-1, keepdims=True)
        e = np.exp(s)
        att = e / e.sum(-1, keepdims=True)
        c["att"] = att
        o = np.einsum("bhts,bshd->bthd", att, v).reshape(B, t, D)
        c["o"] = o
        x = x + o @ p[f"wo{l}"]
        c["xmid"] = x
        b, c["r2"] = rmsnorm_f(x, p[f"g2{l}"])
        c["b"] = b
        gate = b @ p[f"wg{l}"]
        up = b @ p[f"wu{l}"]
        sg = sigmoid(gate)
        act = gate * sg * up
        c["gate"], c["up"], c["sg"], c["act"] = gate, up, sg, act
        x = x + act @ p[f"wd{l}"]
        cache[l] = c
    f, rf = rmsnorm_f(x, p["gf"])
    cache["xf"], cache["f"], cache["rf"] = x, f, rf
    logits = f @ p["lm"]
    return (logits, cache) if want_cache else logits


def loss_and_grads(p, idx, mask, cos, sin):
    B, t = idx.shape
    logits, c = forward(p, idx, cos, sin, want_cache=True)
    lg = logits[:, :-1]
    tgt = idx[:, 1:]
    lg = lg - lg.max(-1, keepdims=True)
    ex = np.exp(lg)
    probs = ex / ex.sum(-1, keepdims=True)
    nm = mask.sum()
    ll = np.log(probs[np.arange(B)[:, None], np.arange(t - 1)[None, :], tgt] + 1e-12)
    loss = -(ll * mask).sum() / nm

    dlg = probs.copy()
    dlg[np.arange(B)[:, None], np.arange(t - 1)[None, :], tgt] -= 1.0
    dlg *= (mask / nm)[..., None]
    dlogits = np.zeros_like(logits)
    dlogits[:, :-1] = dlg

    g = {k: np.zeros_like(v) for k, v in p.items()}
    g["lm"] = c["f"].reshape(-1, D).T @ dlogits.reshape(-1, V)
    df = dlogits @ p["lm"].T
    dx, g["gf"] = rmsnorm_b(df, c["xf"], p["gf"], c["rf"])

    for l in reversed(range(L)):
        cc = c[l]
        # mlp branch
        dact = dx @ p[f"wd{l}"].T
        g[f"wd{l}"] = cc["act"].reshape(-1, MLP).T @ dx.reshape(-1, D)
        dgate = dact * cc["up"] * cc["sg"] * (1 + cc["gate"] * (1 - cc["sg"]))
        dup = dact * cc["gate"] * cc["sg"]
        g[f"wg{l}"] = cc["b"].reshape(-1, D).T @ dgate.reshape(-1, MLP)
        g[f"wu{l}"] = cc["b"].reshape(-1, D).T @ dup.reshape(-1, MLP)
        db = dgate @ p[f"wg{l}"].T + dup @ p[f"wu{l}"].T
        dxm, g[f"g2{l}"] = rmsnorm_b(db, cc["xmid"], p[f"g2{l}"], cc["r2"])
        dx = dx + dxm
        # attention branch
        do = dx @ p[f"wo{l}"].T
        g[f"wo{l}"] = cc["o"].reshape(-1, D).T @ dx.reshape(-1, D)
        do = do.reshape(B, t, HEADS, HD)
        datt = np.einsum("bthd,bshd->bhts", do, cc["v"])
        dv = np.einsum("bhts,bthd->bshd", cc["att"], do)
        ds = cc["att"] * (datt - (datt * cc["att"]).sum(-1, keepdims=True))
        ds /= math.sqrt(HD)
        dqr = np.einsum("bhts,bshd->bthd", ds, cc["kr"])
        dkr = np.einsum("bhts,bthd->bshd", ds, cc["qr"])
        dq = rope_back(dqr, cos, sin).reshape(B, t, D)
        dk = rope_back(dkr, cos, sin).reshape(B, t, D)
        dv = dv.reshape(B, t, D)
        g[f"wq{l}"] = cc["a"].reshape(-1, D).T @ dq.reshape(-1, D)
        g[f"wk{l}"] = cc["a"].reshape(-1, D).T @ dk.reshape(-1, D)
        g[f"wv{l}"] = cc["a"].reshape(-1, D).T @ dv.reshape(-1, D)
        da = dq @ p[f"wq{l}"].T + dk @ p[f"wk{l}"].T + dv @ p[f"wv{l}"].T
        dxi, g[f"g1{l}"] = rmsnorm_b(da, cc["xin"], p[f"g1{l}"], cc["r1"])
        dx = dx + dxi

    np.add.at(g["emb"], c["idx"], dx)
    return loss, g


# --- grad check -------------------------------------------------------------
# tiny config, a few random entries per tensor, finite differences.

def gradcheck():
    global T, D, HEADS, HD, MLP, V, L
    keep = (T, D, HEADS, HD, MLP, V, L)
    T2, D2, H2, HD2, M2, V2, L2 = 7, 8, 2, 4, 12, 13, 2
    T, D, HEADS, HD, MLP, V, L = T2, D2, H2, HD2, M2, V2, L2
    cos, sin = rope_tables(T2, HD2)
    p = init(v=V2, d=D2, mlp=M2)
    idx = rng.integers(0, V2, (2, T2))
    mask = np.ones((2, T2 - 1))
    _, g = loss_and_grads(p, idx, mask, cos, sin)
    worst = 0.0
    for name, w in p.items():
        flat = w.reshape(-1)
        for _ in range(6):
            i = rng.integers(0, flat.size)
            h = 1e-5
            old = flat[i]
            flat[i] = old + h
            lp, _ = loss_and_grads(p, idx, mask, cos, sin)
            flat[i] = old - h
            lm, _ = loss_and_grads(p, idx, mask, cos, sin)
            flat[i] = old
            num = (lp - lm) / (2 * h)
            ana = g[name].reshape(-1)[i]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
    T, D, HEADS, HD, MLP, V, L = keep
    print(f"gradcheck worst rel err {worst:.2e}")
    assert worst < 1e-4, "backward pass is lying"


gradcheck()


# --- train ------------------------------------------------------------------

p = init()
n_params = sum(v.size for v in p.values())
print(f"parameters: {n_params}")

m = {k: np.zeros_like(v) for k, v in p.items()}
vv = {k: np.zeros_like(v) for k, v in p.items()}
b1, b2, aeps = 0.9, 0.999, 1e-8

for step in range(1, STEPS + 1):
    lr = 0.5 * LR * (1 + math.cos(math.pi * step / STEPS)) + 3e-4
    rows = rng.integers(0, len(X), BATCH)
    loss, g = loss_and_grads(p, X[rows], M[rows], COS, SIN)
    for k in p:
        m[k] = b1 * m[k] + (1 - b1) * g[k]
        vv[k] = b2 * vv[k] + (1 - b2) * g[k] * g[k]
        mh = m[k] / (1 - b1 ** step)
        vh = vv[k] / (1 - b2 ** step)
        upd = mh / (np.sqrt(vh) + aeps)
        if p[k].ndim == 2:
            upd = upd + WD * p[k]
        p[k] -= lr * upd
    if step % 500 == 0 or step == 1:
        print(f"step {step:5d}  loss {loss:.4f}")


# --- int8 quantization ------------------------------------------------------
# symmetric, per output channel. embeddings get per dimension scales.

def quant(w, axis):
    s = np.abs(w).max(axis=axis, keepdims=True) / 127.0
    s = np.maximum(s, 1e-8)
    q = np.clip(np.round(w / s), -127, 127).astype(int)
    return q, np.squeeze(s, axis=axis)


Q = {}
S = {}
for name, w in p.items():
    if w.ndim == 2:
        Q[name], S[name] = quant(w, axis=0)  # scale per output column
Q["emb"], S["emb"] = quant(p["emb"], axis=0)  # scale per dim across vocab


def deq(name):
    return Q[name] * S[name][None, :]


# --- reference sampler, css math exactly ------------------------------------
# dequantized weights, plain python floats, same softmax shape as the sheet.

DQ = {k: deq(k) for k in Q}
P_JIT = [13 + 17 * k for k in range(V)]


def sample(qwords, salt, tamp, max_new=GEN):
    ids = encode(qwords)
    s = salt
    out = []
    kcache = [[None] * L for _ in range(T)]
    vcache = [[None] * L for _ in range(T)]

    def step(pos, tok):
        x = DQ["emb"][tok].copy()
        for l in range(L):
            a, _ = rmsnorm_f(x[None, None], p[f"g1{l}"])
            a = a[0, 0]
            q = (a @ DQ[f"wq{l}"]).reshape(HEADS, HD)
            k = (a @ DQ[f"wk{l}"]).reshape(HEADS, HD)
            v = (a @ DQ[f"wv{l}"]).reshape(HEADS, HD)
            for h in range(HEADS):
                for j in range(HD // 2):
                    cq, sq = COS[pos, j], SIN[pos, j]
                    qe, qo = q[h, 2 * j], q[h, 2 * j + 1]
                    ke, ko = k[h, 2 * j], k[h, 2 * j + 1]
                    q[h, 2 * j], q[h, 2 * j + 1] = qe * cq - qo * sq, qe * sq + qo * cq
                    k[h, 2 * j], k[h, 2 * j + 1] = ke * cq - ko * sq, ke * sq + ko * cq
            kcache[pos][l], vcache[pos][l] = k, v
            o = np.zeros(D)
            for h in range(HEADS):
                sc = [float(q[h] @ kcache[i][l][h]) / math.sqrt(HD) for i in range(pos + 1)]
                mx = max(sc)
                es = [math.exp(z - mx) for z in sc]
                tot = sum(es)
                acc = np.zeros(HD)
                for i in range(pos + 1):
                    acc += (es[i] / tot) * vcache[i][l][h]
                o[h * HD : (h + 1) * HD] = acc
            x = x + o @ DQ[f"wo{l}"]
            b, _ = rmsnorm_f(x[None, None], p[f"g2{l}"])
            b = b[0, 0]
            gate = b @ DQ[f"wg{l}"]
            up = b @ DQ[f"wu{l}"]
            x = x + ((gate * sigmoid(gate)) * up) @ DQ[f"wd{l}"]
        f, _ = rmsnorm_f(x[None, None], p["gf"])
        return f[0, 0] @ DQ["lm"]

    for pos, tok in enumerate(ids):
        logits = step(pos, tok)
    cur = None
    for n in range(max_new):
        if cur is not None:
            logits = step(PROMPT + n - 1, cur)
        lj = [logits[k] + k * 1e-5 + tamp * (((s * P_JIT[k]) % 97) / 96 - 0.5) for k in range(V)]
        cur = max(range(V), key=lambda k: lj[k])
        s = (s * 137 + 29) % 251
        out.append(cur)
    return out


def detok(ids):
    ws = []
    for i in ids:
        if i == END:
            break
        ws.append(vocab[i])
    return " ".join(ws)


PREVIEW = [
    "hello", "who are you ?", "how do you work ?", "do you like treats ?",
    "tell me about your day", "are you a real llm ?", "do you use javascript ?", "good girl",
]
print("\n--- int8 reference samples ---")
for tamp in (0.0, 1.0, 2.0):
    print(f"\ntamp {tamp}")
    for qtext in PREVIEW:
        ids = sample(qtext.split(), 3, tamp)
        print(f"  [{qtext}] {detok(ids)}")

out = {
    "vocab": vocab, "T": T, "PROMPT": PROMPT, "D": D, "L": L, "HEADS": HEADS,
    "HD": HD, "MLP": MLP, "rope_base": ROPE_BASE, "eps": EPS, "P": P_JIT,
    "n_params": int(n_params),
    "cos": np.round(COS, 8).tolist(), "sin": np.round(SIN, 8).tolist(),
    "Q": {k: v.tolist() for k, v in Q.items()},
    "S": {k: np.round(v, 8).tolist() for k, v in S.items()},
    "norms": {k: np.round(p[k], 6).tolist() for k in p if p[k].ndim == 1},
}
with open("weights.json", "w") as f:
    json.dump(out, f)
print("\nsaved weights.json")
