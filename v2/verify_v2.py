#!/usr/bin/env python3
"""Reference inference and structural checks for qwennie v2's exported int8 checkpoint."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def rms(x, g, eps):
    return x / math.sqrt(float(np.mean(x * x)) + eps) * g


def deq(q, s):
    return np.asarray(q, dtype=np.float64) * np.asarray(s, dtype=np.float64)[None, :]


def rope(v, pos, cos, sin):
    y = v.copy()
    for r in range(v.shape[-1] // 2):
        e, o = 2 * r, 2 * r + 1
        c, s = cos[pos][r], sin[pos][r]
        ve, vo = v[e], v[o]
        y[e], y[o] = ve * c - vo * s, ve * s + vo * c
    return y


class Ref:
    def __init__(self, path):
        self.w = json.loads(Path(path).read_text(encoding="utf-8"))
        self.vocab = self.w["vocab"]
        self.stoi = {x: i for i, x in enumerate(self.vocab)}
        self.c = self.w["config"]
        self.N = {k: np.asarray(v, dtype=np.float64) for k, v in self.w["norms"].items()}
        self.W = {k: deq(self.w["Q"][k], self.w["S"][k]) for k in self.w["Q"]}
        self.cos = np.asarray(self.w["cos"])
        self.sin = np.asarray(self.w["sin"])
        self.ak = self.w["allowed_keys"]
        c = self.c
        self.D, self.L, self.QH, self.HD = c["D"], c["L"], c["Q_HEADS"], c["HD"]
        self.T, self.SLOTS, self.A1, self.A2 = c["T"], c["SLOTS"], c["A1"], c["A2"]
        self.G, self.eps = c["sample_group"], c["eps"]
        self.PAD, self.USR, self.BOT, self.END = 0, 1, 2, 3

    def prompt(self, text):
        ws = text.split()[: self.SLOTS]
        ids = [self.stoi.get(w, self.PAD) for w in ws]
        ids += [self.PAD] * (self.SLOTS - len(ids))
        return [self.USR] + ids + [self.BOT]

    def step(self, pos, tok, kcache, vcache):
        x = self.W["emb"][tok].copy()
        for l in range(self.L):
            a = rms(x, self.N[f"g1{l}"], self.eps)
            q = (a @ self.W[f"wq{l}"]).reshape(self.QH, self.HD)
            k = a @ self.W[f"wk{l}"]
            v = a @ self.W[f"wv{l}"]
            qr = np.stack([rope(q[h], pos, self.cos, self.sin) for h in range(self.QH)])
            kr = rope(k, pos, self.cos, self.sin)
            kcache[l][pos] = kr.copy()
            vcache[l][pos] = v.copy()
            out = np.zeros(self.D)
            keys = self.ak[pos]
            for h in range(self.QH):
                scores = np.asarray([float(qr[h] @ kcache[l][i]) / math.sqrt(self.HD) for i in keys])
                scores -= scores.max()
                ex = np.exp(scores)
                att = ex / ex.sum()
                o = sum(att[n] * vcache[l][i] for n, i in enumerate(keys))
                out[h * self.HD : (h + 1) * self.HD] = o
            x = x + out @ self.W[f"wo{l}"]
            b = rms(x, self.N[f"g2{l}"], self.eps)
            gate = b @ self.W[f"wg{l}"]
            up = b @ self.W[f"wu{l}"]
            x = x + (gate / (1 + np.exp(-gate)) * up) @ self.W[f"wd{l}"]
        return rms(x, self.N["gf"], self.eps) @ self.W["lm"]

    @staticmethod
    def lcg(s):
        return (s * 137 + 29) % 251

    def sample(self, logits, seed, temp):
        z = np.exp((logits - logits.max()) / temp)
        groups = [z[i : i + self.G].sum() for i in range(0, len(z), self.G)]
        s2 = self.lcg(seed)
        u1 = (s2 + 0.5) / 251
        s3 = self.lcg(s2)
        u2 = (s3 + 0.5) / 251
        gcdf = np.cumsum(groups) / sum(groups)
        g = min(int(np.searchsorted(gcdf, u1, side="right")), len(groups) - 1)
        lo = g * self.G
        zz = z[lo : min(len(z), lo + self.G)]
        j = min(int(np.searchsorted(np.cumsum(zz) / zz.sum(), u2, side="right")), len(zz) - 1)
        return lo + j, s3

    def detok(self, ids):
        out = ""
        for i in ids:
            t = self.vocab[i]
            out += t if t in (".", "!", "?", ",", ":", ";") else (" " if out else "") + t
        return out

    def run_chat(self, q1, q2, rootseed=17, temp=0.8):
        kc = [[None] * self.T for _ in range(self.L)]
        vc = [[None] * self.T for _ in range(self.L)]
        seq, r1, r2 = [], [], []
        seed = rootseed
        for tok in self.prompt(q1):
            pos = len(seq)
            logits = self.step(pos, tok, kc, vc)
            seq.append(tok)
        done = False
        for _ in range(self.A1):
            tok, seed = self.sample(logits, seed, temp)
            tok = self.END if done else tok
            if tok == self.END:
                done = True
            else:
                r1.append(tok)
            pos = len(seq)
            seq.append(tok)
            logits = self.step(pos, tok, kc, vc)
        seed = (rootseed * 193 + 17) % 251
        for tok in self.prompt(q2):
            pos = len(seq)
            logits = self.step(pos, tok, kc, vc)
            seq.append(tok)
        done = False
        for n in range(self.A2):
            tok, seed = self.sample(logits, seed, temp)
            tok = self.END if done else tok
            if tok == self.END:
                done = True
            else:
                r2.append(tok)
            pos = len(seq)
            seq.append(tok)
            if n < self.A2 - 1 and pos < self.T - 1:
                logits = self.step(pos, tok, kc, vc)
        assert len(seq) == self.T
        return self.detok(r1), self.detok(r2), seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "weights_v2.json"))
    ap.add_argument("--expected", default=str(HERE / "expected_v2.json"))
    args = ap.parse_args()
    r = Ref(args.weights)

    # Structural gates: these should fail if a future refactor accidentally regresses
    # the CSS-native constraints that make v2 worthwhile.
    kv_props = r.c["L"] * (r.c["T"] - 1) * r.c["KV_DIM"] * 2
    assert r.w["n_params_dense"] >= 190_000
    assert r.c["KV_HEADS"] == 1 and r.c["Q_HEADS"] >= 4
    assert r.c["T"] >= 64 and r.c["LOCAL_WINDOW"] <= 8
    assert kv_props < 7_000, kv_props
    eligible = r.w.get("sparse_eligible_total", 0)
    eligible_nz = r.w.get("sparse_eligible_nonzero", eligible)
    if eligible:
        assert eligible_nz / eligible <= 0.56, (eligible_nz, eligible)

    samples = [
        ("who are you ?", "how do you work ?"),
        ("do you like treats ?", "what kind ?"),
        ("good girl", "who is good ?"),
        ("what is attention ?", "all words ?"),
        ("remember snow", "what did i say ?"),
        ("can we chat twice ?", "now what ?"),
    ]
    combos = {}
    for seed in (17, 53):
        for temp in (0.45, 0.8, 1.35):
            for q1, q2 in samples:
                a1, a2, seq = r.run_chat(q1, q2, seed, temp)
                key = f"{seed}|{temp}|{q1}|{q2}"
                combos[key] = {"a1": a1, "a2": a2, "ids": seq}
                print(f"[{seed} {temp}] {q1} -> {a1} // {q2} -> {a2}")

    Path(args.expected).write_text(json.dumps({"combos": combos}, indent=2), encoding="utf-8")
    print(f"PASS: {len(combos)} quantized reference chats; kv properties={kv_props:,}")


if __name__ == "__main__":
    main()
