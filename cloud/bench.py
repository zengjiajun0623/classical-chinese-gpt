import time, numpy as np, torch, torch.nn as nn
from torch.nn import functional as F
device="cuda"; torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
class Attn(nn.Module):
    def __init__(s,d,h): super().__init__(); s.qkv=nn.Linear(d,3*d); s.proj=nn.Linear(d,d); s.h=h
    def forward(s,x):
        B,T,C=x.shape; q,k,v=s.qkv(x).split(C,2)
        q=q.view(B,T,s.h,C//s.h).transpose(1,2);k=k.view(B,T,s.h,C//s.h).transpose(1,2);v=v.view(B,T,s.h,C//s.h).transpose(1,2)
        return s.proj(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(B,T,C))
class Blk(nn.Module):
    def __init__(s,d,h): super().__init__(); s.l1=nn.LayerNorm(d);s.a=Attn(d,h);s.l2=nn.LayerNorm(d);s.f=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): x=x+s.a(s.l1(x)); x=x+s.f(s.l2(x)); return x
class G(nn.Module):
    def __init__(s,L,d,h,V=32000,bs=512): super().__init__(); s.t=nn.Embedding(V,d);s.p=nn.Embedding(bs,d);s.b=nn.ModuleList([Blk(d,h) for _ in range(L)]);s.ln=nn.LayerNorm(d);s.hd=nn.Linear(d,V,bias=False);s.hd.weight=s.t.weight
    def forward(s,idx,tg):
        B,T=idx.shape;x=s.t(idx)+s.p(torch.arange(T,device=idx.device))
        for bl in s.b: x=bl(x)
        return F.cross_entropy(s.hd(s.ln(x)).view(-1,32000),tg.view(-1))
data=np.memmap("/workspace/build/data.bin",dtype=np.uint16,mode="r")
def batch(bs,block):
    ix=torch.randint(len(data)-block-1,(bs,))
    x=torch.stack([torch.from_numpy(data[i:i+block].astype(np.int64)) for i in ix])
    y=torch.stack([torch.from_numpy(data[i+1:i+1+block].astype(np.int64)) for i in ix])
    return x.to(device),y.to(device)
def bench(name,L,d,h,bs,block=512,steps=12):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m=G(L,d,h).to(device); opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    P=sum(p.numel() for p in m.parameters())/1e6
    for _ in range(3):
        x,y=batch(bs,block)
        with torch.autocast("cuda",dtype=torch.bfloat16): loss=m(x,y)
        loss.backward();opt.step();opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize();t=time.time()
    for _ in range(steps):
        x,y=batch(bs,block)
        with torch.autocast("cuda",dtype=torch.bfloat16): loss=m(x,y)
        loss.backward();opt.step();opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize();dt=time.time()-t
    print(f"{name}: {P:.0f}M params | bs {bs} | {bs*block*steps/dt:,.0f} tok/s | {torch.cuda.max_memory_allocated()/1e9:.1f} GB peak",flush=True)
    del m,opt; torch.cuda.empty_cache()
for args in [("124M-small",12,768,12,24),("355M-medium",24,1024,16,10),("470M-large",24,1152,18,8)]:
    try: bench(*args)
    except Exception as e: print(args[0],"FAILED:",str(e)[:120],flush=True); torch.cuda.empty_cache()
