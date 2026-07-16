"""Production-shape RAG: retrieval over ALL of Chinese Wikipedia.

Pipeline per question:
  1. search zh.wikipedia.org for candidate pages (their search backend)
  2. fetch each candidate's lead paragraph (simplified Chinese via Accept-Language)
  3. rank candidates by character-bigram overlap with the question (similarity)
  4. hand the best passage to our 124M model as the 参考 card; it answers from it

The model is unchanged (gpt2_zh_rag.pt). Only the library grew: 30 hand cards -> 1.4M articles.
"""
import sys, json, urllib.request, urllib.parse, torch, torch.nn as nn
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

import time
UA={"User-Agent":"chinese-gpt-from-scratch/1.0 (learning project)","Accept-Language":"zh-cn"}
def http(url):
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers=UA)
            return json.load(urllib.request.urlopen(req,timeout=15))
        except urllib.error.HTTPError as e:
            if e.code==429 and attempt<2: time.sleep(6); continue   # rate-limited: back off
            raise

def clean(q):
    for junk in ["谁是","什么是","告诉我更多关于","的信息","告诉我","？","?","。"]:
        q=q.replace(junk,"")
    return q.strip() or q

def wiki_search(q,limit=5):
    u=("https://zh.wikipedia.org/w/api.php?action=query&list=search&format=json"
       "&srlimit=%d&srsearch=%s" % (limit,urllib.parse.quote(clean(q))))
    try: return [r["title"] for r in http(u)["query"]["search"]]
    except Exception as e:
        print("  [search error] %r" % e, flush=True); return []

def wiki_lead(title):
    u="https://zh.wikipedia.org/api/rest_v1/page/summary/"+urllib.parse.quote(title)
    try:
        d=http(u); return d.get("extract","")
    except Exception: return ""

def bigrams(s): return {s[i:i+2] for i in range(len(s)-1)}
def sim(q,passage):
    a,b=bigrams(q),bigrams(passage[:80])
    return len(a&b)/max(1,len(a))

def retrieve(q):
    """search -> fetch leads -> rank by similarity to the question -> best passage"""
    key=clean(q); cands=[]
    for t in wiki_search(q):
        lead=wiki_lead(t)
        if not lead: continue
        score=sim(q,t+lead)
        if t==key or t.replace("習","习")==key: score+=2.0   # exact-title match wins
        elif key and key in t: score+=0.5
        score-=len(t)*0.01                                   # tie-break: shorter title
        cands.append((score,t,lead))
    if not cands: return None,None
    cands.sort(key=lambda c:-c[0])
    _,t,lead=cands[0]
    return t,lead[:220]

@torch.no_grad()
def ask(q,n=110,rep=1.3):
    title,ctx=retrieve(q)
    card=ctx if ctx else "无"
    ids=tk.encode("参考："+card+"\n问："+q+"\n答：").ids; start=len(ids)
    idx=torch.tensor([ids],device=device)
    for _ in range(n):
        lg=m(idx[:,-block_size:])[:,-1,:]
        # penalize only tokens already GENERATED (not the reference: copying stays free)
        for t in set(idx[0,start:].tolist()):
            lg[0,t]=lg[0,t]/rep if lg[0,t]>0 else lg[0,t]*rep
        nx=lg.argmax(dim=-1,keepdim=True)
        if nx.item()==EOT: break
        idx=torch.cat([idx,nx],dim=1)
    return title,card,tk.decode(idx[0,start:].tolist())

if __name__=="__main__":
    QS=["谁是习近平？","谁是爱因斯坦？","谁是马云？","谁是李世民？","什么是黑洞？"]
    for q in QS:
        t,c,a=ask(q)
        print("问："+q)
        print("〔检索〕维基百科条目「"+(t or "无")+"」")
        print("〔片段〕"+(c or "无"))
        print("答："+a+"\n", flush=True)
        time.sleep(3)   # politeness: stay under the rate limit
