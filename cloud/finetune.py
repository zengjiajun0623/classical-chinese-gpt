import time, numpy as np, torch, torch.nn as nn
from torch.nn import functional as F
OUT="/workspace/build"; device="cuda"
block_size=512; batch_size=32; n_layer,n_head,n_embd=12,12,768; vocab_size=32000
lr=1.5e-4; max_iters=700; eval_interval=100; P_CLASSICAL=0.7
torch.manual_seed(1337); torch.backends.cuda.matmul.allow_tf32=True
class Attn(nn.Module):
    def __init__(s): super().__init__(); s.qkv=nn.Linear(n_embd,3*n_embd); s.proj=nn.Linear(n_embd,n_embd)
    def forward(s,x):
        B,T,C=x.shape; q,k,v=s.qkv(x).split(n_embd,dim=2)
        q=q.view(B,T,n_head,C//n_head).transpose(1,2);k=k.view(B,T,n_head,C//n_head).transpose(1,2);v=v.view(B,T,n_head,C//n_head).transpose(1,2)
        return s.proj(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(B,T,C))
class MLP(nn.Module):
    def __init__(s): super().__init__(); s.fc=nn.Linear(n_embd,4*n_embd); s.proj=nn.Linear(4*n_embd,n_embd)
    def forward(s,x): return s.proj(F.gelu(s.fc(x)))
class Block(nn.Module):
    def __init__(s): super().__init__(); s.ln1=nn.LayerNorm(n_embd); s.attn=Attn(); s.ln2=nn.LayerNorm(n_embd); s.mlp=MLP()
    def forward(s,x): x=x+s.attn(s.ln1(x)); x=x+s.mlp(s.ln2(x)); return x
class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok=nn.Embedding(vocab_size,n_embd); s.pos=nn.Embedding(block_size,n_embd)
        s.blocks=nn.ModuleList([Block() for _ in range(n_layer)]); s.lnf=nn.LayerNorm(n_embd)
        s.head=nn.Linear(n_embd,vocab_size,bias=False); s.head.weight=s.tok.weight
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
ck=torch.load(OUT+"/gpt2_zh.pt",map_location=device)
model=GPT().to(device); model.load_state_dict(ck["model"]); print("loaded base ckpt",flush=True)
data_c=np.memmap(OUT+"/classical.bin",dtype=np.uint16,mode="r"); Nc=len(data_c)
data_w=np.memmap(OUT+"/data.bin",dtype=np.uint16,mode="r"); Nw=len(data_w)
print(f"classical {Nc:,} | wiki {Nw:,} | blend {int(P_CLASSICAL*100)}/{int((1-P_CLASSICAL)*100)}",flush=True)
def get_batch():
    xs=[]; ys=[]
    for _ in range(batch_size):
        if torch.rand(1).item()<P_CLASSICAL: d,N=data_c,Nc
        else: d,N=data_w,Nw
        i=torch.randint(N-block_size-1,(1,)).item()
        xs.append(torch.from_numpy(d[i:i+block_size].astype(np.int64)))
        ys.append(torch.from_numpy(d[i+1:i+1+block_size].astype(np.int64)))
    return torch.stack(xs).to(device), torch.stack(ys).to(device)
from tokenizers import ByteLevelBPETokenizer
tk=ByteLevelBPETokenizer(OUT+"/tok/zhbpe-vocab.json",OUT+"/tok/zhbpe-merges.txt")
opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.1,betas=(0.9,0.95))
t0=time.time()
for it in range(max_iters+1):
    if it%eval_interval==0:
        model.eval(); idx=torch.tensor([tk.encode("话说").ids],device=device)
        with torch.autocast("cuda",dtype=torch.bfloat16): out=model.gen(idx,120)
        print(f"iter {it} {time.time()-t0:.0f}s | {tk.decode(out[0].tolist()).replace(chr(10),' ')[:180]}",flush=True)
        model.train(); torch.save({"model":model.state_dict(),"cfg":ck["cfg"]},OUT+"/gpt2_zh_classical.pt")
    if it==max_iters: break
    x,y=get_batch()
    with torch.autocast("cuda",dtype=torch.bfloat16): _,loss=model(x,y)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True)
print("FINETUNE DONE",flush=True)
