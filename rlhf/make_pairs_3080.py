import sys, json, torch, torch.nn as nn
from torch.nn import functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
D=r"C:\Users\Jiajun\cgpt"; device="cuda"
block_size=512; n_layer,n_head,n_embd=12,12,768; vocab_size=32000
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
    def forward(s,idx):
        B,T=idx.shape; x=s.tok(idx)+s.pos(torch.arange(T,device=idx.device))
        for b in s.blocks: x=b(x)
        return s.head(s.lnf(x))
ck=torch.load(D+r"\gpt2_zh_classical.pt", map_location=device)
m=GPT().to(device); m.load_state_dict(ck["model"]); m.eval()
from tokenizers import ByteLevelBPETokenizer
tk=ByteLevelBPETokenizer(D+r"\tok\zhbpe-vocab.json", D+r"\tok\zhbpe-merges.txt")
@torch.no_grad()
def gen(prompt,n=90,temp=0.95,topk=120,rep=1.3):
    idx=torch.tensor([tk.encode(prompt).ids],device=device); start=idx.shape[1]
    for _ in range(n):
        lg=m(idx[:,-block_size:])[:,-1,:]
        for t in set(idx[0].tolist()):
            lg[0,t]=lg[0,t]/rep if lg[0,t]>0 else lg[0,t]*rep
        lg=lg/temp
        v,_=torch.topk(lg,topk); lg[lg<v[:,[-1]]]=-float("inf")
        idx=torch.cat([idx,torch.multinomial(F.softmax(lg,dim=-1),1)],dim=1)
    return tk.decode(idx[0].tolist()).replace(chr(10)," ")
prompts=["话说那员外","宝玉正说着","那道人手持宝剑","一日，众好汉","却说那婆子","黛玉含泪道"]
pairs=[]
for i,p in enumerate(prompts):
    a=gen(p); b=gen(p)
    pairs.append({"prompt":p,"a":a,"b":b})
    print("\n===== Pair %d | seed: %s =====" % (i+1,p))
    print("A) "+a)
    print("B) "+b)
json.dump(pairs, open(D+r"\pairs.json","w",encoding="utf-8"), ensure_ascii=False)
print("\n(saved %d pairs to pairs.json)" % len(pairs))
