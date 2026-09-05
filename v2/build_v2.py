#!/usr/bin/env python3
"""Compile a qwennie v2 checkpoint into a pure-CSS two-turn chat page."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fmt(x):
    r = repr(float(x))
    return "0" if r in ("0.0", "-0.0") else r


def css_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def glyph_text(tok):
    if tok in ("<p>", "<u>", "<b>", "<e>"):
        return ""
    if tok in (".", "!", "?", ",", ":", ";"):
        return tok
    return " " + tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "weights_v2.json"))
    ap.add_argument("--outdir", default=str(HERE / "site"))
    args = ap.parse_args()
    w = json.loads(Path(args.weights).read_text(encoding="utf-8"))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    vocab = w["vocab"]; V = len(vocab); stoi = {x:i for i,x in enumerate(vocab)}
    c = w["config"]
    T=c["T"]; SLOTS=c["SLOTS"]; P1=c["P1"]; A1=c["A1"]; P2=c["P2"]; A2=c["A2"]
    B1=c["TURN1_BOT"]; T2S=c["TURN2_START"]; B2=c["TURN2_BOT"]
    D=c["D"]; L=c["L"]; QH=c["Q_HEADS"]; KVH=c["KV_HEADS"]; HD=c["HD"]; KVD=c["KV_DIM"]; MLP=c["MLP"]
    EPS=c["eps"]; GROUP=c["sample_group"]
    EBITS=c["emb_bits"]; ECMOD=c["emb_code_mod"]
    ECMUL=c["emb_code_mul"]; ECADD=c["emb_code_add"]; ESCALE=c["emb_scale"]
    PAD,USR,BOT,END=0,1,2,3
    Q=w["Q"]; S=w["S"]; N=w["norms"]; COS=w["cos"]; SIN=w["sin"]; AK=w["allowed_keys"]
    SQ=math.sqrt(HD)

    # All CSS matmuls use int coefficients * one per-output scale.
    css=[]; reg=[]
    def prop(name, syntax="<number>", inherits=False, initial="0"):
        reg.append(f'@property {name} {{ syntax: "{syntax}"; inherits: {"true" if inherits else "false"}; initial-value: {initial}; }}')

    css += ["/* qwennie v2 — generated. pure CSS transformer runtime. */"]
    for name,syntax,initial in [
        ("--tk","<integer>","0"),("--np","<integer>","0"),("--sd","<number>","17"),
        ("--rootseed","<number>","17"),("--tamp","<number>","0.8"),("--dn","<number>","0"),
        ("--dnext","<number>","0"),("--snext","<number>","17"),("--u1","<number>","0.5"),("--u2","<number>","0.5")]:
        prop(name,syntax,True,initial)
    prop("--probe","<number>",False,"0")
    prop("--ecode","<number>",False,"0")
    for b in range(EBITS): prop(f"--eb{b}","<number>",False,"0")

    # Working registers. Prefix per layer to keep one shared .cl rule.
    def lp(l): return f"l{l}_"
    for l in range(L):
        p=lp(l)
        for nm,n in [("x",D),("a",D),("q",D),("k",KVD),("v",KVD),("qr",D),("kr",KVD),("o",D),("om",D),("xm",D),("b",D),("gt",MLP),("up",MLP),("ac",MLP),("xn",D)]:
            for j in range(n): prop(f"--{p}{nm}{j}")
        prop(f"--{p}r1"); prop(f"--{p}r2")
    for j in range(D): prop(f"--f{j}")
    prop("--rf")
    for k in range(V):
        prop(f"--lg{k}"); prop(f"--ev{k}")
    prop("--lmx"); prop("--zt"); prop("--gsel")
    groups=math.ceil(V/GROUP)
    for g in range(groups):
        prop(f"--gm{g}"); prop(f"--gcf{g}"); prop(f"--pick{g}")
    # score/sum scratch: one name per layer/head/key-position is enough across query positions.
    for l in range(L):
        for h in range(QH):
            for i in range(T-1): prop(f"--s{l}h{h}i{i}")
            prop(f"--mx{l}h{h}"); prop(f"--den{l}h{h}")
    # Shared MQA cache: dramatically smaller than v1's full-D per-head cache.
    for l in range(L):
        for pos in range(T-1):
            for j in range(KVD):
                prop(f"--K{l}p{pos}j{j}", inherits=True)
                prop(f"--V{l}p{pos}j{j}", inherits=True)
    css.append("\n".join(reg)); css.append("")

    # Input and controls.
    css.append(".chat{--rootseed:17;--sd:17;--tamp:.8;--dn:0}")
    seeds=[("ball",17,"🎾"),("bone",53,"🦴"),("duck",101,"🦆"),("zoom",199,"💨")]
    temps=[("cozy",.45),("waggy",.8),("zoomies",1.35)]
    for name,seed,_ in seeds:
        css.append(f".page:has(#seed-{name}:checked) .chat{{--rootseed:{seed};--sd:{seed}}}")
    for name,temp in temps:
        css.append(f".page:has(#temp-{name}:checked) .chat{{--tamp:{fmt(temp)}}}")

    # Prompt slots are backed by <select>; option:checked drives the token id.
    input_ids=w.get("input_token_ids") or [i for i,t in enumerate(vocab) if i>=4]
    for turn in (1,2):
        for slot in range(SLOTS):
            pos=(1+slot) if turn==1 else (T2S+1+slot)
            cls=f"u{turn}s{slot}w"
            # blank option -> PAD
            css.append(f".page:has(#u{turn}s{slot} option[value=\"{PAD}\"]:checked) .{cls}{{--tk:{PAD}}}")
            for tid in input_ids:
                css.append(f".page:has(#u{turn}s{slot} option[value=\"{tid}\"]:checked) .{cls}{{--tk:{tid}}}")
    css.append("")

    # Embeddings deliberately avoid @container style(--tk: ...) lookups. Chromium
    # only propagates a short chain when a queried property depends on an earlier
    # style-query result. v2 uses an invertible 10-bit token code instead, so the
    # embedding is plain arithmetic in the ordinary custom-property DAG.

    def matmul(dst, src, name, nin, nout, indent="  "):
        lines=[]; q=Q[name]; s=S[name]
        for j in range(nout):
            terms=[]
            for i in range(nin):
                coef=q[i][j]
                if coef:
                    terms.append(("+" if coef>0 else "-") + f" var(--{src}{i})*{abs(coef)}")
            body="0 " + " ".join(terms)
            lines.append(f"{indent}--{dst}{j}:calc({fmt(s[j])}*({body}));")
        return lines

    def rms(rname, src, n):
        sq=" + ".join(f"var(--{src}{j})*var(--{src}{j})" for j in range(n))
        return f"  --{rname}:calc(1/sqrt(({sq})/{n} + {fmt(EPS)}));"

    cell=[".cl{"]
    # Binary token embedding. Scramble tk into a unique 10-bit code, expose each
    # bit once, then form every embedding dimension from ten learned bit deltas.
    prop_bits = []
    cell.append(f"  --ecode:mod(var(--tk)*{ECMUL} + {ECADD}, {ECMOD});")
    for b in range(EBITS):
        shift=1<<b
        if b == 0:
            expr="mod(var(--ecode), 2)"
        else:
            expr=f"mod((var(--ecode) - mod(var(--ecode), {shift}))/{shift}, 2)"
        cell.append(f"  --eb{b}:{expr};")
    qtab=Q["emb_bits"]; stab=S["emb_bits"]
    for j in range(D):
        base=0.0
        terms=[]
        for b in range(EBITS):
            v0=qtab[2*b][j]*stab[j]
            v1=qtab[2*b+1][j]*stab[j]
            base += v0
            delta=(v1-v0)
            if delta:
                terms.append(("+" if delta > 0 else "-") + f" {fmt(abs(ESCALE*delta))}*var(--eb{b})")
        base *= ESCALE
        cell.append(f"  --l0_x{j}:calc({fmt(base)} {' '.join(terms)});")
    for l in range(L):
        p=lp(l); xin=f"{p}x"
        if l>0:
            # copy previous layer output into this layer input
            prev=lp(l-1)
            for j in range(D): cell.append(f"  --{p}x{j}:var(--{prev}xn{j});")
        cell.append(rms(f"{p}r1",xin,D))
        for j in range(D): cell.append(f"  --{p}a{j}:calc(var(--{xin}{j})*var(--{p}r1)*{fmt(N[f'g1{l}'][j])});")
        cell += matmul(f"{p}q",f"{p}a",f"wq{l}",D,D)
        cell += matmul(f"{p}k",f"{p}a",f"wk{l}",D,KVD)
        cell += matmul(f"{p}v",f"{p}a",f"wv{l}",D,KVD)
        # attention output --o is positional; projection is shared.
        cell += matmul(f"{p}om",f"{p}o",f"wo{l}",D,D)
        for j in range(D): cell.append(f"  --{p}xm{j}:calc(var(--{xin}{j}) + var(--{p}om{j}));")
        cell.append(rms(f"{p}r2",f"{p}xm",D))
        for j in range(D): cell.append(f"  --{p}b{j}:calc(var(--{p}xm{j})*var(--{p}r2)*{fmt(N[f'g2{l}'][j])});")
        cell += matmul(f"{p}gt",f"{p}b",f"wg{l}",D,MLP)
        cell += matmul(f"{p}up",f"{p}b",f"wu{l}",D,MLP)
        for j in range(MLP):
            # SiLU(gate) * up
            cell.append(f"  --{p}ac{j}:calc((var(--{p}gt{j})/(1 + exp(-1*var(--{p}gt{j}))))*var(--{p}up{j}));")
        # down projection + residual, manually to fold the add.
        qd=Q[f"wd{l}"]; sd=S[f"wd{l}"]
        for j in range(D):
            terms=[]
            for i in range(MLP):
                coef=qd[i][j]
                if coef: terms.append(("+" if coef>0 else "-") + f" var(--{p}ac{i})*{abs(coef)}")
            cell.append(f"  --{p}xn{j}:calc(var(--{p}xm{j}) + {fmt(sd[j])}*(0 {' '.join(terms)}));")
    last=lp(L-1)
    cell.append(rms("rf",f"{last}xn",D))
    for j in range(D): cell.append(f"  --f{j}:calc(var(--{last}xn{j})*var(--rf)*{fmt(N['gf'][j])});")
    cell.append("}")
    css.append("\n".join(cell)); css.append("")

    # Position-specific RoPE, cache writes, and sparse local-global attention.
    lev=[]
    for pos in range(T-1):
        lines=[f".cl.p{pos}{{"]
        keys=AK[pos]
        for l in range(L):
            p=lp(l)
            # Q has Q_HEADS heads.
            for h in range(QH):
                for r in range(HD//2):
                    co,si=fmt(COS[pos][r]),fmt(SIN[pos][r]); e=h*HD+2*r; o=e+1
                    lines.append(f"--{p}qr{e}:calc(var(--{p}q{e})*{co} - var(--{p}q{o})*{si});")
                    lines.append(f"--{p}qr{o}:calc(var(--{p}q{e})*{si} + var(--{p}q{o})*{co});")
            # one shared K head
            for r in range(HD//2):
                co,si=fmt(COS[pos][r]),fmt(SIN[pos][r]); e=2*r; o=e+1
                lines.append(f"--{p}kr{e}:calc(var(--{p}k{e})*{co} - var(--{p}k{o})*{si});")
                lines.append(f"--{p}kr{o}:calc(var(--{p}k{e})*{si} + var(--{p}k{o})*{co});")
            for j in range(KVD):
                lines.append(f"--K{l}p{pos}j{j}:var(--{p}kr{j});--V{l}p{pos}j{j}:var(--{p}v{j});")
            for h in range(QH):
                for i in keys:
                    dot=" + ".join(f"var(--{p}qr{h*HD+j})*" + (f"var(--{p}kr{j})" if i==pos else f"var(--K{l}p{i}j{j})") for j in range(HD))
                    lines.append(f"--s{l}h{h}i{i}:calc(({dot})/{fmt(SQ)});")
                mx=",".join(f"var(--s{l}h{h}i{i})" for i in keys)
                lines.append(f"--mx{l}h{h}:max({mx});")
                den=" + ".join(f"exp(var(--s{l}h{h}i{i}) - var(--mx{l}h{h}))" for i in keys)
                lines.append(f"--den{l}h{h}:calc({den});")
                for j in range(HD):
                    num=" + ".join(f"exp(var(--s{l}h{h}i{i}) - var(--mx{l}h{h}))*" + (f"var(--{p}v{j})" if i==pos else f"var(--V{l}p{i}j{j})") for i in keys)
                    lines.append(f"--{p}o{h*HD+j}:calc(({num})/var(--den{l}h{h}));")
        lines.append("}")
        lev.append("\n".join(lines))
    css.append("\n".join(lev)); css.append("")

    # LM head and exact hierarchical categorical sampler.
    gc=[".gc{"]
    qlm=Q["lm"]; slm=S["lm"]
    for k in range(V):
        terms=[]
        for j in range(D):
            coef=qlm[j][k]
            if coef: terms.append(("+" if coef>0 else "-") + f" var(--f{j})*{abs(coef)}")
        gc.append(f"--lg{k}:calc({fmt(slm[k])}*(0 {' '.join(terms)}));")
    gc.append("--lmx:max("+",".join(f"var(--lg{k})" for k in range(V))+");")
    for k in range(V): gc.append(f"--ev{k}:exp((var(--lg{k}) - var(--lmx))/var(--tamp));")
    for g in range(groups):
        ids=list(range(g*GROUP,min(V,(g+1)*GROUP)))
        gc.append(f"--gm{g}:calc("+" + ".join(f"var(--ev{k})" for k in ids)+");")
    gc.append("--zt:calc("+" + ".join(f"var(--gm{g})" for g in range(groups))+");")
    gc.append("--u1:calc((mod(var(--sd)*137 + 29,251) + .5)/251);")
    gc.append("--u2:calc((mod(mod(var(--sd)*137 + 29,251)*137 + 29,251) + .5)/251);")
    # direct cumulative sums => shallow dependency graph
    for g in range(groups):
        gc.append(f"--gcf{g}:calc(("+" + ".join(f"var(--gm{x})" for x in range(g+1))+f")/var(--zt));")
    if groups>1:
        gc.append("--gsel:calc("+" + ".join(f"max(0,sign(var(--u1) - var(--gcf{g})))" for g in range(groups-1))+");")
    else: gc.append("--gsel:0;")
    for g in range(groups):
        ids=list(range(g*GROUP,min(V,(g+1)*GROUP)))
        comps=[]
        for x in range(len(ids)-1):
            cum=" + ".join(f"var(--ev{k})" for k in ids[:x+1])
            comps.append(f"max(0,sign(var(--u2) - (({cum})/var(--gm{g}))))")
        offset=ids[0]
        gc.append(f"--pick{g}:calc({offset}" + (" + " + " + ".join(comps) if comps else "") + ");")
    gc.append("--np:calc("+" + ".join(f"var(--pick{g})*max(0,1 - abs(var(--gsel) - {g}))" for g in range(groups))+");")
    gc.append("--snext:mod(mod(var(--sd)*137 + 29,251)*137 + 29,251);")
    gc.append(f"--dnext:max(var(--dn),calc(1 - min(1,abs(var(--np) - {END}))));")
    gc.append("}")
    css.append("\n".join(gc)); css.append("")

    # Generated-token wrapper: consume sampler result; after END force END forever.
    css.append(f".gw{{--dn:var(--dnext);--sd:var(--snext);--tk:calc(var(--dn)*{END} + (1 - var(--dn))*var(--np));}}")
    # Turn 2 deliberately resets done + derives a fresh deterministic substream from root seed.
    css.append(".turn2reset{--dn:0;--sd:mod(var(--rootseed)*193 + 17,251)}")
    # Visible token lookup without style queries. CSS counter styles are an exact
    # integer -> symbol table, which is precisely what detokenization needs.
    symbols=" ".join(f'"{css_escape(glyph_text(tok))}"' for tok in vocab)
    css.append(f'@counter-style qwennie-vocab{{system:fixed 0;symbols:{symbols};fallback:decimal}}')
    css.append('.g{counter-reset:qwtok var(--tk)}')
    css.append('.g::before{content:counter(qwtok,qwennie-vocab)}')
    css.append("")

    # Feature probe.
    css.append(".page{--probe:calc(max(0,sign(mod(3,2))) + 0*sqrt(4) + 0*exp(0))}")
    css.append("@container style(--probe: 1){.warn{display:none}}")
    model_css="\n".join(css)
    (outdir/"model_v2.css").write_text(model_css,encoding="utf-8")

    # ---------- HTML tree ----------
    # Build from final token backwards. Prompt wrapper owns token; .cl computes logits at that token.
    def gen_wrapper(pos, inner, with_gc, turn, extra=""):
        cls=f"cl p{pos}"+(" gc" if with_gc else "")
        wcls=f"w gw gen t{turn}" + (f" {extra}" if extra else "")
        return f'<span class="{wcls}"><span class="g"></span><span class="{cls}">{inner}</span></span>'
    def fixed_wrapper(pos, token, inner, extra=""):
        return f'<span class="w {extra}" style="--tk:{token}"><span class="cl p{pos}">{inner}</span></span>'
    def slot_wrapper(pos, turn, slot, inner, reset=False):
        extra=f"u{turn}s{slot}w" + (" turn2reset" if reset else "")
        return f'<span class="w {extra}"><span class="cl p{pos}">{inner}</span></span>'

    # final token position 63 is generated by pos62; only glyph needed.
    tree='<span class="w gw gen t2"><span class="g"></span></span>'
    # turn2 generated positions 62 down to 36; each pos<63 computes next token
    for pos in range(T-2, B2, -1):
        tree=gen_wrapper(pos,tree,True,2,"turn2first" if pos==B2+1 else "")
    # BOT2 must sample first reply2 token.
    tree=f'<span class="w bot2" style="--tk:{BOT}"><span class="cl p{B2} gc">{tree}</span></span>'
    # second prompt slots, backwards. reset on first causal token (USR), not slot.
    for slot in reversed(range(SLOTS)):
        pos=T2S+1+slot
        tree=slot_wrapper(pos,2,slot,tree)
    tree=f'<span class="w turn2reset" style="--tk:{USR}"><span class="cl p{T2S}">{tree}</span></span>'
    # visible transcript marker at the causal boundary; it carries no model state.
    tree='<span class="turnbreak">qwennie · turn 2</span>'+tree
    # turn1 generated positions 25..10: pos25 has no gc because next token is fixed USR2
    for pos in range(B1+A1, B1, -1):
        with_gc = pos < B1+A1
        tree=gen_wrapper(pos,tree,with_gc,1)
    # The loop above starts at 25 with no gc and goes to10.
    # BOT1 samples first turn1 token.
    tree=f'<span class="w bot1" style="--tk:{BOT}"><span class="cl p{B1} gc">{tree}</span></span>'
    for slot in reversed(range(SLOTS)):
        pos=1+slot
        tree=slot_wrapper(pos,1,slot,tree)
    tree=f'<span class="w" style="--tk:{USR}"><span class="cl p0">{tree}</span></span>'

    # controls use a modest input lexicon; blank PAD first.
    def options(default_words, turn, slot):
        default=default_words[slot] if slot<len(default_words) else None
        op=[f'<option value="{PAD}"'+(" selected" if default is None else "")+'>·</option>']
        for tid in input_ids:
            tok=vocab[tid]
            sel=" selected" if tok==default else ""
            op.append(f'<option value="{tid}"{sel}>{css_escape(tok)}</option>')
        return "".join(op)
    d1=["who","are","you","?"]; d2=["how","do","you","work","?"]
    selects1="".join(f'<select id="u1s{s}" aria-label="turn 1 token {s+1}">{options(d1,1,s)}</select>' for s in range(SLOTS))
    selects2="".join(f'<select id="u2s{s}" aria-label="turn 2 token {s+1}">{options(d2,2,s)}</select>' for s in range(SLOTS))
    seed_controls="".join(f'<input type="radio" name="seed" id="seed-{n}"'+(" checked" if n=="ball" else "")+f'><label for="seed-{n}">{ico}</label>' for n,_,ico in seeds)
    temp_controls="".join(f'<input type="radio" name="temp" id="temp-{n}"'+(" checked" if n=="waggy" else "")+f'><label for="temp-{n}">{n}</label>' for n,_ in temps)

    style='''
:root{font-family:ui-rounded,system-ui,sans-serif;color-scheme:dark;--bg:#111015;--card:#1b1822;--ink:#f8eefb;--muted:#bcaec4;--acc:#ff93d0}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#26182b,var(--bg) 42%);color:var(--ink);min-height:100vh}.page{max-width:980px;margin:auto;padding:48px 20px 80px}h1{font-size:clamp(42px,9vw,82px);margin:0;letter-spacing:-.06em}.tag{color:var(--muted);max-width:760px;line-height:1.55}.panel{background:color-mix(in srgb,var(--card) 92%,transparent);border:1px solid #3a3040;border-radius:22px;padding:18px;margin:18px 0;box-shadow:0 18px 50px #0005}.turn{display:grid;gap:10px}.who{font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:var(--muted)}.slots{display:flex;gap:6px;flex-wrap:wrap}.slots select{background:#121017;color:var(--ink);border:1px solid #44374b;border-radius:10px;padding:8px;max-width:130px}.reply{font-size:19px;line-height:1.55;min-height:34px}.reply:before{content:'qwennie ';color:var(--acc);font-weight:700}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.controls input{position:absolute;opacity:0;pointer-events:none}.controls label{padding:7px 11px;border:1px solid #44374b;border-radius:999px;cursor:pointer}.controls input:checked+label{background:var(--acc);color:#261220;border-color:var(--acc)}.warn{background:#462436;border:1px solid #80435f;padding:12px 14px;border-radius:12px}.brain{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}.meta{color:var(--muted);font-size:13px}.paw{font-size:.55em;vertical-align:top}
'''
    # Tree emits glyphs in document order. CSS hides the brain, so duplicate visible glyph collectors are impossible.
    # Instead make generated .g visible by taking them out of the clipped brain using position:fixed would be bad.
    # We place the causal tree in the two reply boxes via CSS display:contents and turn classes.
    style += '.brain{position:static;width:auto;height:auto;overflow:visible;clip-path:none;display:contents}.brain .w,.brain .cl{display:contents}.brain .g{display:none}.brain .g.t{} .reply1 .g,.reply2 .g{display:inline}'
    # Since one nested tree cannot physically live in two reply divs, use generated content collectors below via anchor containers is impossible without JS.
    # So render tree once as a transcript and use CSS labels around turn boundaries.
    style += '.transcript{background:var(--card);border:1px solid #3a3040;border-radius:22px;padding:18px;margin-top:16px}.brain .gen.t1>.g,.brain .gen.t2>.g{display:inline}.brain .gen.t2:first-of-type>.g:before{}.turnbreak{display:block;color:var(--acc);font-weight:700;margin:14px 0 4px}'

    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>qwennie v2 — pure CSS two-turn transformer</title><style>{style}</style><link rel="stylesheet" href="model_v2.css"></head><body><div class="page">
<h1>qwennie <span class="paw">🐩</span> v2</h1><p class="tag">a {w['n_params_dense']:,}-parameter, 3-layer decoder transformer running in the CSS style engine. MQA, local+global causal attention, RoPE, RMSNorm, SwiGLU, sparse int8 matmuls, and exact categorical temperature sampling. still zero JavaScript.</p>
<div class="warn">this browser is missing CSS math/style-query features qwennie needs. Chromium is the intended engine.</div>
<div class="panel controls"><span class="who">reroll</span>{seed_controls}<span class="who">temperature</span>{temp_controls}</div>
<div class="panel turn"><span class="who">you · turn 1</span><div class="slots">{selects1}</div></div>
<div class="panel turn"><span class="who">you · turn 2</span><div class="slots">{selects2}</div></div>
<div class="transcript"><div class="who">qwennie · turn 1 · both replies generated in one style pass</div><div class="brain chat">{tree}</div></div>
<p class="meta">dense params {w['n_params_dense']:,} · quantized nonzero weights {w['quant_nonzero']:,}/{w['quant_total']:,} · vocab {V} · context {T} · {QH}Q/{KVH}KV heads · local window {c['LOCAL_WINDOW']} + memory anchors</p>
</div></body></html>'''
    (outdir/"index.html").write_text(html,encoding="utf-8")
    manifest={"css_bytes":len(model_css.encode()),"vocab":V,"groups":groups,"kv_properties":L*(T-1)*KVD*2,"dense_params":w["n_params_dense"],"quant_nonzero":w["quant_nonzero"],"quant_total":w["quant_total"],"sparse_eligible_nonzero":w.get("sparse_eligible_nonzero"),"sparse_eligible_total":w.get("sparse_eligible_total")}
    (outdir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()
