#!/usr/bin/env python3
"""Train qwennie v2: a CSS-native tiny MQA/local-global decoder transformer.

PyTorch is used for the research/training loop; export is plain JSON and build_v2.py
has no torch dependency. Determinism is intentional. The exported sparse checkpoint
is what the stylesheet executes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent


def split_words(s: str) -> list[str]:
    return s.strip().split() if s.strip() else []


def load_base(path: Path) -> list[tuple[list[str], list[str]]]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "||" not in raw:
            continue
        q, a = [x.strip() for x in raw.split("||", 1)]
        out.append((split_words(q), split_words(a)))
    return out


def load_extra(path: Path) -> list[tuple[list[str], list[str], list[str], list[str]]]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = [x.strip() for x in raw.split("||")]
        if len(parts) != 4:
            raise ValueError(f"expected 4 fields: {raw}")
        out.append(tuple(split_words(x) for x in parts))
    return out


def make_records(base, extra):
    records = []
    # Keep every v1 exchange as a first-turn lesson. Second turn is blank and masked.
    for q, a in base:
        records.append((q, a, [], []))
    # True two-turn records get oversampled so turn-two behavior is not drowned out.
    for x in extra:
        records.extend([x, x, x])
    return records


def build_vocab(records):
    words = sorted({w for rec in records for seq in rec for w in seq})
    vocab = SPECIALS + words
    if len(vocab) > MAX_VOCAB:
        raise ValueError(f"vocab {len(vocab)} > MAX_VOCAB {MAX_VOCAB}")
    return vocab


def pad_prompt(words, stoi):
    if len(words) > SLOTS:
        words = words[-SLOTS:]
    toks = [stoi[w] for w in words] + [PAD] * (SLOTS - len(words))
    return [USR] + toks + [BOT]


def fit_answer(words, max_len, stoi):
    # reserve one slot for END
    ids = [stoi[w] for w in words[: max_len - 1]] + [END]
    return ids + [END] * (max_len - len(ids))


def encode_record(rec, stoi):
    q1, a1, q2, a2 = rec
    seq = pad_prompt(q1, stoi) + fit_answer(a1, A1, stoi)
    if q2:
        seq += pad_prompt(q2, stoi) + fit_answer(a2, A2, stoi)
    else:
        seq += [PAD] * (P2 + A2)
    assert len(seq) == T

    mask = [0.0] * (T - 1)
    # targets for first reply: BOT -> first word through first END
    a1_target_count = min(len(a1), A1 - 1) + 1
    start = TURN1_BOT
    for i in range(a1_target_count):
        mask[start + i] = 1.0
    if q2:
        a2_target_count = min(len(a2), A2 - 1) + 1
        start2 = TURN2_BOT
        for i in range(a2_target_count):
            if start2 + i < T - 1:
                mask[start2 + i] = 1.0
    return seq, mask


def allowed_keys(pos: int) -> list[int]:
    """CSS-native sparse causal pattern.

    - ordinary positions: local window + all completed BOT memory anchors
    - BOT1: full first prompt
    - BOT2: BOT1 + all of assistant1 and user2, becoming a conversation summary anchor
    """
    if pos == TURN1_BOT:
        return list(range(0, pos + 1))
    if pos == TURN2_BOT:
        return sorted(set([TURN1_BOT] + list(range(TURN1_BOT + 1, pos + 1))))
    lo = max(0, pos - LOCAL_WINDOW + 1)
    keys = set(range(lo, pos + 1))
    if pos > TURN1_BOT:
        keys.add(TURN1_BOT)
    if pos > TURN2_BOT:
        keys.add(TURN2_BOT)
    return sorted(k for k in keys if k <= pos)


def make_mask(device):
    m = torch.full((T, T), float("-inf"), device=device)
    for q in range(T):
        m[q, allowed_keys(q)] = 0.0
    return m


def rope_tables(device, dtype=torch.float32):
    inv = ROPE_BASE ** (-torch.arange(HD // 2, device=device, dtype=dtype) / (HD // 2))
    ang = torch.arange(T, device=device, dtype=dtype)[:, None] * inv[None, :]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    # x [B,T,H,HD]
    xe, xo = x[..., 0::2], x[..., 1::2]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    y = torch.empty_like(x)
    y[..., 0::2] = xe * c - xo * s
    y[..., 1::2] = xe * s + xo * c
    return y


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * self.weight


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = RMSNorm(D)
        self.wq = nn.Linear(D, D, bias=False)
        self.wk = nn.Linear(D, KV_DIM, bias=False)
        self.wv = nn.Linear(D, KV_DIM, bias=False)
        self.wo = nn.Linear(D, D, bias=False)
        self.n2 = RMSNorm(D)
        self.wg = nn.Linear(D, MLP, bias=False)
        self.wu = nn.Linear(D, MLP, bias=False)
        self.wd = nn.Linear(MLP, D, bias=False)

    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        a = self.n1(x)
        q = self.wq(a).view(b, t, Q_HEADS, HD)
        k = self.wk(a).view(b, t, KV_HEADS, HD)
        v = self.wv(a).view(b, t, KV_HEADS, HD)
        q = apply_rope(q, cos[:t], sin[:t])
        k = apply_rope(k, cos[:t], sin[:t])
        # MQA: one KV head is shared by all Q heads.
        k0 = k[:, :, 0, :]
        v0 = v[:, :, 0, :]
        score = torch.einsum("bthd,bsd->bhts", q, k0) / math.sqrt(HD)
        score = score + mask[:t, :t][None, None, :, :]
        att = torch.softmax(score, dim=-1)
        o = torch.einsum("bhts,bsd->bthd", att, v0).reshape(b, t, D)
        x = x + self.wo(o)
        z = self.n2(x)
        x = x + self.wd(F.silu(self.wg(z)) * self.wu(z))
        return x


class BinaryTokenEmbedding(nn.Module):
    """Unique 10-bit token code -> learned sum of per-bit state vectors."""
    def __init__(self, vocab_size):
        super().__init__()
        assert vocab_size <= EMB_CODE_MOD
        self.table = nn.Embedding(EMB_BITS * 2, D)
        self.register_buffer("bit_offsets", (torch.arange(EMB_BITS) * 2).long(), persistent=False)

    def forward(self, idx):
        code = torch.remainder(idx * EMB_CODE_MUL + EMB_CODE_ADD, EMB_CODE_MOD)
        shifts = (1 << torch.arange(EMB_BITS, device=idx.device, dtype=idx.dtype))
        bits = torch.remainder(torch.div(code[..., None], shifts, rounding_mode="floor"), 2)
        rows = bits + self.bit_offsets
        return self.table(rows).sum(dim=-2) * EMB_SCALE


class QwennieV2(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = BinaryTokenEmbedding(vocab_size)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.nf = RMSNorm(D)
        self.lm = nn.Linear(D, vocab_size, bias=False)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, idx, cos, sin, mask):
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        return self.lm(self.nf(x))


def structured_2of4(w: np.ndarray) -> np.ndarray:
    """Keep the largest N magnitudes in every M-wide chunk along input dimension."""
    out = w.copy()
    # Export matrices are [in,out]. Group along input axis for each output channel.
    for j in range(out.shape[1]):
        for i in range(0, out.shape[0], SPARSITY_M):
            sl = out[i:i+SPARSITY_M, j]
            if sl.size <= SPARSITY_N:
                continue
            keep = np.argpartition(np.abs(sl), -SPARSITY_N)[-SPARSITY_N:]
            mask = np.zeros(sl.size, dtype=bool); mask[keep] = True
            sl[~mask] = 0.0
            out[i:i+SPARSITY_M, j] = sl
    return out


def quantize_per_out(w: np.ndarray):
    s = np.max(np.abs(w), axis=0, keepdims=True) / 127.0
    s = np.maximum(s, 1e-8)
    q = np.clip(np.round(w / s), -127, 127).astype(np.int16)
    return q, s.squeeze(0)


def export(model, vocab, path, input_token_ids, sparse=True):
    state = model.state_dict()
    mats = {}
    norms = {}
    # Embedding codebooks are [rows,D]. Linear weights are converted from
    # PyTorch [out,in] to compiler-friendly [in,out].
    mats["emb_bits"] = state["emb.table.weight"].cpu().numpy()
    mats["lm"] = state["lm.weight"].cpu().numpy().T  # [D,V]
    norms["gf"] = state["nf.weight"].cpu().numpy()
    for l in range(L):
        p = f"blocks.{l}."
        norms[f"g1{l}"] = state[p+"n1.weight"].cpu().numpy()
        norms[f"g2{l}"] = state[p+"n2.weight"].cpu().numpy()
        for name in ("wq","wk","wv","wo","wg","wu","wd"):
            mats[f"{name}{l}"] = state[p+name+".weight"].cpu().numpy().T
    dense_params = sum(x.numel() for x in model.parameters())
    qdict, sdict = {}, {}
    nonzero = 0; total = 0; eligible_nonzero = 0; eligible_total = 0
    for name, w in mats.items():
        ww = w.copy()
        # Do not sparsify embedding/lm; output quality per byte is better there.
        if sparse and not name.startswith("emb_") and name != "lm":
            ww = structured_2of4(ww)
        q, s = quantize_per_out(ww)
        qdict[name] = q.tolist(); sdict[name] = np.round(s, 9).tolist()
        nz = int(np.count_nonzero(q)); nonzero += nz; total += int(q.size)
        if not name.startswith("emb_") and name != "lm":
            eligible_nonzero += nz; eligible_total += int(q.size)
    cos, sin = rope_tables(torch.device("cpu"))
    payload = {
        "format": "qwennie-v2-css-native-v1",
        "vocab": vocab,
        "config": {
            "T":T,"SLOTS":SLOTS,"P1":P1,"A1":A1,"P2":P2,"A2":A2,
            "TURN1_BOT":TURN1_BOT,"TURN2_START":TURN2_START,"TURN2_BOT":TURN2_BOT,
            "D":D,"L":L,"Q_HEADS":Q_HEADS,"KV_HEADS":KV_HEADS,"HD":HD,"KV_DIM":KV_DIM,
            "MLP":MLP,"LOCAL_WINDOW":LOCAL_WINDOW,"rope_base":ROPE_BASE,"eps":EPS,
            "sample_group":SAMPLE_GROUP,
            "emb_bits":EMB_BITS,"emb_code_mod":EMB_CODE_MOD,
            "emb_code_mul":EMB_CODE_MUL,"emb_code_add":EMB_CODE_ADD,
            "emb_scale":EMB_SCALE,
        },
        "n_params_dense": int(dense_params),
        "quant_nonzero": nonzero,
        "quant_total": total,
        "sparse_eligible_nonzero": eligible_nonzero,
        "sparse_eligible_total": eligible_total,
        "Q": qdict,
        "S": sdict,
        "norms": {k: np.round(v, 7).tolist() for k,v in norms.items()},
        "cos": np.round(cos.numpy(), 9).tolist(),
        "sin": np.round(sin.numpy(), 9).tolist(),
        "allowed_keys": [allowed_keys(i) for i in range(T)],
        "input_token_ids": input_token_ids,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"exported {path}: {dense_params:,} dense params, {nonzero:,}/{total:,} nonzero quant weights")


@torch.no_grad()
def enforce_2of4(model):
    """Project trainable linear weights onto 2:4 sparsity along each input row-group."""
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or name == "lm":
            continue
        w = mod.weight  # [out,in]
        out_dim, in_dim = w.shape
        for start in range(0, in_dim, SPARSITY_M):
            width = min(SPARSITY_M, in_dim - start)
            if width <= SPARSITY_N:
                continue
            block = w[:, start:start+width]
            keep = block.abs().topk(SPARSITY_N, dim=1).indices
            mask = torch.zeros_like(block)
            mask.scatter_(1, keep, 1.0)
            block.mul_(mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", default=str(HERE / "weights_v2.json"))
    ap.add_argument("--dense-export", action="store_true", help="disable 2:4 sparsity in exported checkpoint")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint", default=str(HERE / "train_v2.pt"))
    ap.add_argument("--max-run-steps", type=int, default=0, help="stop after this many optimizer steps, saving a resumable checkpoint")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(5, os.cpu_count() or 1)))
    base_path = ROOT / "corpus.txt"
    if not base_path.exists():
        base_path = HERE / "corpus_smoke.txt"
    base = load_base(base_path)
    extra = load_extra(HERE / "extra_v2.txt")
    records = make_records(base, extra)
    vocab = build_vocab(records); stoi = {w:i for i,w in enumerate(vocab)}
    from collections import Counter
    qfreq = Counter(w for q1,_,q2,_ in records for w in (q1 + q2))
    input_words = [w for w,_ in qfreq.most_common(128)]
    input_token_ids = [stoi[w] for w in input_words]
    encoded = [encode_record(r, stoi) for r in records]
    X = torch.tensor([x for x,_ in encoded], dtype=torch.long)
    M = torch.tensor([m for _,m in encoded], dtype=torch.float32)
    print(f"records={len(records)} vocab={len(vocab)} shape={tuple(X.shape)}")

    device = torch.device("cpu")
    model = QwennieV2(len(vocab)).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"dense parameters={nparams:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.999), weight_decay=WD)
    cos, sin = rope_tables(device)
    amask = make_mask(device)
    gen = torch.Generator().manual_seed(SEED + 1)
    start_step = 0
    ckpt_path = Path(args.checkpoint)
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_step = int(ck["step"])
        if "gen_state" in ck: gen.set_state(ck["gen_state"])
        print(f"resumed {ckpt_path} at step {start_step}")

    model.train()
    end_step = args.steps if not args.max_run_steps else min(args.steps, start_step + args.max_run_steps)
    for step in range(start_step + 1, end_step + 1):
        idx = torch.randint(0, len(X), (BATCH,), generator=gen)
        xb, mb = X[idx].to(device), M[idx].to(device)
        logits = model(xb, cos, sin, amask)
        lg = logits[:, :-1, :].reshape(-1, len(vocab))
        tgt = xb[:, 1:].reshape(-1)
        mm = mb.reshape(-1)
        ce = F.cross_entropy(lg, tgt, reduction="none")
        loss = (ce * mm).sum() / mm.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        frac = step / max(1, args.steps)
        lr = MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups: g["lr"] = lr
        opt.step()
        if not args.dense_export and step >= max(1, int(args.steps * 0.60)):
            enforce_2of4(model)
        if step == 1 or step % max(25, args.steps // 20) == 0:
            print(f"step {step:5d}/{args.steps} loss={loss.item():.4f} lr={lr:.6f}", flush=True)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": end_step, "gen_state": gen.get_state()}, ckpt_path)
    print(f"saved checkpoint {ckpt_path} at step {end_step}", flush=True)
    if end_step >= args.steps:
        export(model, vocab, Path(args.out), input_token_ids, sparse=not args.dense_export)
    else:
        print("training chunk complete; resume for the remaining steps", flush=True)

if __name__ == "__main__":
    main()
