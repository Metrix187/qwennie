#!/usr/bin/env python3
"""Reference inference and structural verification for qwennie v2's exported int8 checkpoint."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def rms(x,g,eps):
    return x / math.sqrt(float(np.mean(x*x))+eps) * g

def rope(v, pos, cos, sin):
    y=v.copy()
    for r in range(v.shape[-1]//2):
        e=2*r; o=e+1; c=cos[pos][r]; s=sin[pos][r]
        ve,vo=v[e],v[o]
        y[e]=ve*c-vo*s; y[o]=ve*s+vo*c
    return y

def deq(q,s): return np.asarray(q,dtype=np.float64)*np.asarray(s,dtype=np.float64)[None,:]

class Ref:
    def __init__(self,path):
        self.w=json.loads(Path(path).read_text())
        self.vocab=self.w['vocab']; self.stoi={x:i for i,x in enumerate(self.vocab)}; self.c=self.w['config']
        self.Q=self.w['Q']; self.S=self.w['S']; self.N={k:np.asarray(v,dtype=np.float64) for k,v in self.w['norms'].items()}
        self.W={k:deq(self.Q[k],self.S[k]) for k in self.Q}
        self.cos=np.asarray(self.w['cos']); self.sin=np.asarray(self.w['sin']); self.ak=self.w['allowed_keys']
        self.D=self.c['D']; self.L=self.c['L']; self.QH=self.c['Q_HEADS']; self.HD=self.c['HD']; self.KVD=self.c['KV_DIM']; self.MLP=self.c['MLP']; self.eps=self.c['eps']
        self.T=self.c['T']; self.SLOTS=self.c['SLOTS']; self.A1=self.c['A1']; self.A2=self.c['A2']; self.B1=self.c['TURN1_BOT']; self.T2S=self.c['TURN2_START']; self.B2=self.c['TURN2_BOT']; self.G=self.c['sample_group']
        self.PAD,self.USR,self.BOT,self.END=0,1,2,3
        self.EBITS=self.c['emb_bits']; self.ECMOD=self.c['emb_code_mod']
        self.ECMUL=self.c['emb_code_mul']; self.ECADD=self.c['emb_code_add']; self.ESCALE=self.c['emb_scale']
    def prompt(self,text):
        ws=text.split()[:self.SLOTS]
        ids=[self.stoi.get(w,self.PAD) for w in ws] + [self.PAD]*(self.SLOTS-len(ws))
        return [self.USR]+ids+[self.BOT]
    def embedding(self,tok):
        code=(tok*self.ECMUL+self.ECADD)%self.ECMOD
        rows=[2*b + ((code>>b)&1) for b in range(self.EBITS)]
        return self.ESCALE*sum((self.W['emb_bits'][r] for r in rows), start=np.zeros(self.D))
    def step(self,pos,tok,kcache,vcache):
        x=self.embedding(tok).copy()
        for l in range(self.L):
            a=rms(x,self.N[f'g1{l}'],self.eps)
            q=(a@self.W[f'wq{l}']).reshape(self.QH,self.HD)
            k=a@self.W[f'wk{l}']; v=a@self.W[f'wv{l}']
            qr=np.stack([rope(q[h],pos,self.cos,self.sin) for h in range(self.QH)])
            kr=rope(k,pos,self.cos,self.sin)
            kcache[l][pos]=kr.copy(); vcache[l][pos]=v.copy()
            out=np.zeros(self.D)
            for h in range(self.QH):
                keys=self.ak[pos]
                scores=np.array([float(qr[h]@kcache[l][i])/math.sqrt(self.HD) for i in keys])
                scores-=scores.max(); ex=np.exp(scores); att=ex/ex.sum()
                o=sum(att[n]*vcache[l][i] for n,i in enumerate(keys))
                out[h*self.HD:(h+1)*self.HD]=o
            x=x+out@self.W[f'wo{l}']
            b=rms(x,self.N[f'g2{l}'],self.eps)
            gate=b@self.W[f'wg{l}']; up=b@self.W[f'wu{l}']
            silu=gate/(1+np.exp(-gate))
            x=x+(silu*up)@self.W[f'wd{l}']
        f=rms(x,self.N['gf'],self.eps)
        return f@self.W['lm']
    @staticmethod
    def lcg(s): return (s*137+29)%251
    def sample(self, logits, seed, temp):
        z=np.exp((logits-logits.max())/temp)
        groups=[z[i:i+self.G].sum() for i in range(0,len(z),self.G)]
        s2=self.lcg(seed); u1=(s2+.5)/251
        s3=self.lcg(s2); u2=(s3+.5)/251
        g=int(np.searchsorted(np.cumsum(groups)/sum(groups),u1,side='right'))
        g=min(g,len(groups)-1); lo=g*self.G; zz=z[lo:min(len(z),lo+self.G)]
        j=int(np.searchsorted(np.cumsum(zz)/zz.sum(),u2,side='right')); j=min(j,len(zz)-1)
        return lo+j,s3
    def run_chat(self,q1,q2,rootseed=17,temp=.8):
        kcache=[[None]*self.T for _ in range(self.L)]; vcache=[[None]*self.T for _ in range(self.L)]
        seq=[]; r1=[]; r2=[]; seed=rootseed
        for tok in self.prompt(q1):
            pos=len(seq); logits=self.step(pos,tok,kcache,vcache); seq.append(tok)
        done=False
        for n in range(self.A1):
            tok,seed=self.sample(logits,seed,temp)
            if done: tok=self.END
            if tok==self.END: done=True
            if not done: r1.append(tok)
            pos=len(seq); seq.append(tok)
            logits=self.step(pos,tok,kcache,vcache)
        seed=(rootseed*193+17)%251
        for tok in self.prompt(q2):
            pos=len(seq); logits=self.step(pos,tok,kcache,vcache); seq.append(tok)
        done=False
        for n in range(self.A2):
            tok,seed=self.sample(logits,seed,temp)
            if done: tok=self.END
            if tok==self.END: done=True
            if not done:r2.append(tok)
            pos=len(seq); seq.append(tok)
            if n < self.A2-1 and pos < self.T-1: logits=self.step(pos,tok,kcache,vcache)
        assert len(seq)==self.T
        return self.detok(r1),self.detok(r2),seq
    def detok(self,ids):
        out=''
        for i in ids:
            t=self.vocab[i]
            if t in ('.','!','?',',',':',';'): out+=t
            else: out+=(' ' if out else '')+t
        return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weights',default=str(HERE/'weights_v2.json')); ap.add_argument('--expected',default=str(HERE/'expected_v2.json')); args=ap.parse_args()
    r=Ref(args.weights)
    samples=[
        ('who are you ?','how do you work ?'),('do you like treats ?','what kind ?'),('good girl','who is good ?'),
        ('what is attention ?','all words ?'),('remember snow','what did i say ?'),('can we chat twice ?','now what ?')]
    combos={}
    for seed in (17,53):
        for temp in (.45,.8,1.35):
            for q1,q2 in samples:
                a1,a2,seq=r.run_chat(q1,q2,seed,temp)
                key=f'{seed}|{temp}|{q1}|{q2}'; combos[key]={'a1':a1,'a2':a2,'ids':seq}
                print(f'[{seed} {temp}] {q1} -> {a1} // {q2} -> {a2}')
    Path(args.expected).write_text(json.dumps({'combos':combos},indent=2),encoding='utf-8')
    assert r.w['n_params_dense'] >= 190_000
    assert r.c['KV_HEADS']==1 and r.c['Q_HEADS']>=4
    kv=r.c['L']*(r.c['T']-1)*r.c['KV_DIM']*2
    assert kv < 7000, kv
    assert len(r.vocab) <= 544
    print(f'\nPASS params={r.w["n_params_dense"]:,} vocab={len(r.vocab)} kv_props={kv} expected={args.expected}')

if __name__=='__main__': main()
