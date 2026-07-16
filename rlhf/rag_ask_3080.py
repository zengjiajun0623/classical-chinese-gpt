"""Talk to the RAG question-answering model.
Retrieves a fact-card by entity match, feeds it to the SFT'd model, prints the grounded answer.
Add entries to knowledge.json to expand what it knows. No retraining needed.
Questions are read from QUESTIONS below (edit and re-run)."""
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
m=GPT().to(device); m.load_state_dict(torch.load(D+r"\gpt2_zh_rag.pt",map_location=device)["model"]); m.eval()
from tokenizers import ByteLevelBPETokenizer
tk=ByteLevelBPETokenizer(D+r"\tok\zhbpe-vocab.json", D+r"\tok\zhbpe-merges.txt")
EOT=tk.token_to_id("<|endoftext|>")
KB=json.load(open(D+r"\knowledge.json",encoding="utf-8"))
def retrieve(q):
    hits=[e for e in KB if e in q]
    return KB[max(hits,key=len)] if hits else "无"
@torch.no_grad()
def ask(q,n=90,temp=0.2,topk=10):
    card=retrieve(q); ids=tk.encode("参考："+card+"\n问："+q+"\n答：").ids; start=len(ids)
    idx=torch.tensor([ids],device=device)
    for _ in range(n):
        lg=m(idx[:,-block_size:])[:,-1,:]/temp
        v,_=torch.topk(lg,topk); lg[lg<v[:,[-1]]]=-float("inf")
        nx=torch.multinomial(F.softmax(lg,dim=-1),1)
        if nx.item()==EOT: break
        idx=torch.cat([idx,nx],dim=1)
    return card, tk.decode(idx[0,start:].tolist())

QUESTIONS=["谁是关羽？","谁是妲己？","武松出自哪部小说？","谁是苏格拉底？"]
for q in QUESTIONS:
    c,a=ask(q)
    print("问："+q); print("  〔查到〕"+c); print("答："+a+"\n")
