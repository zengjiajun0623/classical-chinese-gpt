import os, time, math, numpy as np, torch, torch.nn as nn
from torch.nn import functional as F

DATA="/workspace/build/data.bin"; OUT="/workspace/build"; device="cuda"
block_size=512; batch_size=32; grad_accum=8
n_layer,n_head,n_embd=12,12,768; vocab_size=32000; dropout=0.0
lr,min_lr=6e-4,6e-5; warmup,max_iters=500,17000
eval_interval,eval_iters,sample_interval=1000,40,1000
weight_decay,grad_clip=0.1,1.0

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True

data=np.memmap(DATA,dtype=np.uint16,mode="r"); N=len(data); split=int(N*0.999)
print(f"tokens {N:,} (train {split:,} / val {N-split:,})", flush=True)
def get_batch(which):
    d=data[:split] if which=="train" else data[split:]
    ix=torch.randint(len(d)-block_size-1,(batch_size,))
    x=torch.stack([torch.from_numpy(d[i:i+block_size].astype(np.int64)) for i in ix])
    y=torch.stack([torch.from_numpy(d[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.pin_memory().to(device,non_blocking=True), y.pin_memory().to(device,non_blocking=True)

class Attn(nn.Module):
    def __init__(s):
        super().__init__(); s.qkv=nn.Linear(n_embd,3*n_embd); s.proj=nn.Linear(n_embd,n_embd)
    def forward(s,x):
        B,T,C=x.shape; q,k,v=s.qkv(x).split(n_embd,dim=2)
        q=q.view(B,T,n_head,C//n_head).transpose(1,2); k=k.view(B,T,n_head,C//n_head).transpose(1,2); v=v.view(B,T,n_head,C//n_head).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return s.proj(y.transpose(1,2).contiguous().view(B,T,C))
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.fc=nn.Linear(n_embd,4*n_embd); s.proj=nn.Linear(4*n_embd,n_embd)
    def forward(s,x): return s.proj(F.gelu(s.fc(x)))
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.ln1=nn.LayerNorm(n_embd); s.attn=Attn(); s.ln2=nn.LayerNorm(n_embd); s.mlp=MLP()
    def forward(s,x): x=x+s.attn(s.ln1(x)); x=x+s.mlp(s.ln2(x)); return x
class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok=nn.Embedding(vocab_size,n_embd); s.pos=nn.Embedding(block_size,n_embd)
        s.blocks=nn.ModuleList([Block() for _ in range(n_layer)]); s.lnf=nn.LayerNorm(n_embd)
        s.head=nn.Linear(n_embd,vocab_size,bias=False); s.head.weight=s.tok.weight
        s.apply(s._init)
    def _init(s,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,0,0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
    def forward(s,idx,targets=None):
        B,T=idx.shape; x=s.tok(idx)+s.pos(torch.arange(T,device=idx.device))
        for b in s.blocks: x=b(x)
        logits=s.head(s.lnf(x)); loss=None
        if targets is not None: loss=F.cross_entropy(logits.view(-1,vocab_size),targets.view(-1))
        return logits,loss
    @torch.no_grad()
    def gen(s,idx,n,temp=0.8,topk=200):
        for _ in range(n):
            lg,_=s(idx[:,-block_size:]); lg=lg[:,-1,:]/temp
            v,_=torch.topk(lg,topk); lg[lg<v[:,[-1]]]=-float("inf")
            idx=torch.cat([idx,torch.multinomial(F.softmax(lg,dim=-1),1)],dim=1)
        return idx

model=GPT().to(device)
print("params(M):",round(sum(p.numel() for p in model.parameters())/1e6,1),flush=True)
from tokenizers import ByteLevelBPETokenizer
tk=ByteLevelBPETokenizer(OUT+"/tok/zhbpe-vocab.json",OUT+"/tok/zhbpe-merges.txt")
opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay,betas=(0.9,0.95))
def get_lr(it):
    if it<warmup: return lr*it/warmup
    if it>max_iters: return min_lr
    r=(it-warmup)/(max_iters-warmup); return min_lr+0.5*(lr-min_lr)*(1+math.cos(math.pi*r))
@torch.no_grad()
def evaluate():
    model.eval(); o={}
    for w in ["train","val"]:
        L=torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y=get_batch(w)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16): _,l=model(x,y)
            L[k]=l.item()
        o[w]=L.mean().item()
    model.train(); return o
t0=time.time()
for it in range(max_iters+1):
    for g in opt.param_groups: g["lr"]=get_lr(it)
    if it%eval_interval==0:
        m=evaluate(); print(f"iter {it}  train {m['train']:.3f}  val {m['val']:.3f}  lr {get_lr(it):.2e}  {time.time()-t0:.0f}s",flush=True)
        torch.save({"model":model.state_dict(),"cfg":dict(n_layer=n_layer,n_head=n_head,n_embd=n_embd,block_size=block_size,vocab_size=vocab_size)},OUT+"/gpt2_zh.pt")
    if it%sample_interval==0:
        model.eval(); idx=torch.tensor([tk.encode("中国").ids],device=device)
        print("  sample:",tk.decode(model.gen(idx,120)[0].tolist()).replace("\n"," ")[:220],flush=True); model.train()
    if it==max_iters: break
    for _ in range(grad_accum):
        x,y=get_batch("train")
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16): _,loss=model(x,y); loss=loss/grad_accum
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip); opt.step(); opt.zero_grad(set_to_none=True)
print("TRAINING DONE",flush=True)
