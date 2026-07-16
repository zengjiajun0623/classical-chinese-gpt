import sys, json, random, torch, torch.nn as nn
from torch.nn import functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
random.seed(0)
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
from tokenizers import ByteLevelBPETokenizer
tk=ByteLevelBPETokenizer(D+r"\tok\zhbpe-vocab.json", D+r"\tok\zhbpe-merges.txt")

data=[]; nh=0; na=0
# your 6 REAL human picks stay permanently in the mix (corrected pair 3)
picks1=["a","b","a","a","b","a"]
for pr,pk in zip(json.load(open(D+r"\pairs.json",encoding="utf-8")),picks1):
    ch=pr[pk]; rj=pr["b" if pk=="a" else "a"]; pl=len(tk.encode(pr["prompt"]).ids)
    data.append((pl,tk.encode(ch).ids[:block_size],tk.encode(rj).ids[:block_size])); nh+=1
# autonomous RLAIF labels
for pr in json.load(open(D+r"\prefs_auto.json",encoding="utf-8")):
    pl=len(tk.encode(pr["prompt"]).ids)
    data.append((pl,tk.encode(pr["chosen"]).ids[:block_size],tk.encode(pr["rejected"]).ids[:block_size])); na+=1
print("training on %d prefs = %d human + %d auto-labeled, on %s" % (len(data),nh,na,torch.cuda.get_device_name(0)), flush=True)

def build():
    m=GPT().to(device); m.load_state_dict(torch.load(D+r"\gpt2_zh_classical.pt",map_location=device)["model"]); return m
policy=build(); ref=build()
for p in ref.parameters(): p.requires_grad=False
ref.eval()
opt=torch.optim.AdamW(policy.parameters(), lr=1e-6); beta=0.1
def seq_logp(model,ids,pl):
    x=torch.tensor([ids[:-1]],device=device); y=torch.tensor([ids[1:]],device=device)
    lp=F.log_softmax(model(x),dim=-1); tl=lp.gather(2,y.unsqueeze(-1)).squeeze(-1)[0]
    mask=(torch.arange(tl.shape[0],device=device)>=(pl-1)).float()
    return (tl*mask).sum()
EPOCHS=5
for ep in range(EPOCHS):
    order=list(range(len(data))); random.shuffle(order); tot=0.0; acc=0; kl=0.0
    for j in order:
        pl,ci,ri=data[j]
        pc=seq_logp(policy,ci,pl); prj=seq_logp(policy,ri,pl)
        with torch.no_grad(): rc=seq_logp(ref,ci,pl); rr=seq_logp(ref,ri,pl)
        margin=beta*((pc-rc)-(prj-rr)); loss=-F.logsigmoid(margin)
        opt.zero_grad(); loss.backward(); opt.step()
        tot+=loss.item(); acc+=1 if margin.item()>0 else 0; kl+=abs((pc-rc).item())/max(1,len(ci))
    print("epoch %d  dpo_loss %.4f  chosen>rejected %d/%d  ref_drift/tok %.4f" % (ep+1,tot/len(data),acc,len(data),kl/len(data)), flush=True)
torch.save({"model":policy.state_dict()}, D+r"\gpt2_zh_rlaif.pt")
print("saved gpt2_zh_rlaif.pt", flush=True)
