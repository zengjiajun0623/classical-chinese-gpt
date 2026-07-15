import torch, torch.nn as nn
from torch.nn import functional as F

device='mps'
block_size, n_embd, n_head, n_layer, dropout = 128, 192, 6, 4, 0.0

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv=nn.Linear(n_embd,3*n_embd,bias=False)
        self.proj=nn.Linear(n_embd,n_embd); self.drop=nn.Dropout(dropout)
    def forward(self,x):
        B,T,C=x.shape
        q,k,v=self.qkv(x).split(n_embd,dim=2)
        q=q.view(B,T,n_head,C//n_head).transpose(1,2)
        k=k.view(B,T,n_head,C//n_head).transpose(1,2)
        v=v.view(B,T,n_head,C//n_head).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.drop(self.proj(y.transpose(1,2).contiguous().view(B,T,C)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_embd,4*n_embd),nn.ReLU(),
                               nn.Linear(4*n_embd,n_embd),nn.Dropout(dropout))
    def forward(self,x): return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa=CausalSelfAttention(); self.ff=FeedForward()
        self.ln1=nn.LayerNorm(n_embd); self.ln2=nn.LayerNorm(n_embd)
    def forward(self,x):
        x=x+self.sa(self.ln1(x)); x=x+self.ff(self.ln2(x)); return x

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb=nn.Embedding(vocab_size,n_embd); self.pos_emb=nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd); self.head=nn.Linear(n_embd,vocab_size)
    def forward(self,idx):
        B,T=idx.shape
        x=self.tok_emb(idx)+self.pos_emb(torch.arange(T,device=device))
        return self.head(self.ln_f(self.blocks(x)))

ckpt=torch.load('model_tags.pt', map_location=device)
stoi=ckpt['stoi']; itos=ckpt['itos']; vocab_size=len(stoi)
encode=lambda s:[stoi[c] for c in s]; decode=lambda l:''.join(itos[i] for i in l)
model=GPT(vocab_size).to(device); model.load_state_dict(ckpt['model']); model.eval()

@torch.no_grad()
def summon(tag, k=90, temperature=0.8):
    idx=torch.tensor([encode('=== %s ===\n' % tag)], device=device)
    ban=stoi['=']
    for _ in range(k):
        logits=model(idx[:,-block_size:])[:,-1,:]/temperature
        logits[:,ban]=float('-inf')          # forbid new tags: style stays locked
        probs=F.softmax(logits,dim=-1)
        idx=torch.cat([idx,torch.multinomial(probs,1)],dim=1)
    return decode(idx[0].tolist()).split('\n',1)[1]

for tag in ['红楼梦','三国演义','聊斋志异','鲁迅']:
    print('\n========== %s ==========' % tag)
    print(summon(tag))
