import torch, torch.nn as nn, time, re
from torch.nn import functional as F

torch.manual_seed(1337)
device = 'mps'
block_size, batch_size = 128, 32
n_embd, n_head, n_layer, dropout = 192, 6, 4, 0.1
lr, max_iters, eval_interval, eval_iters = 3e-4, 1500, 250, 20

train_text = open('train.txt', encoding='utf-8').read()
val_text   = open('val.txt',   encoding='utf-8').read()
chars = sorted(set(train_text + val_text)); vocab_size = len(chars)
stoi = {c:i for i,c in enumerate(chars)}; itos = {i:c for i,c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]; decode = lambda l: ''.join(itos[i] for i in l)
train_data = torch.tensor(encode(train_text), dtype=torch.long, device=device)
val_data   = torch.tensor(encode(val_text),   dtype=torch.long, device=device)
print('vocab:', vocab_size, ' train:', len(train_data), ' val:', len(val_data))

span = torch.arange(block_size, device=device)
def get_batch(split):
    d = train_data if split=='train' else val_data
    ix = torch.randint(len(d)-block_size-1, (batch_size,), device=device)
    idx = ix[:,None] + span
    return d[idx], d[idx+1]

@torch.no_grad()
def estimate_loss():
    out={}; model.eval()
    for sp in ['train','val']:
        L=torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y=get_batch(sp); _,loss=model(x,y); L[k]=loss.item()
        out[sp]=L.mean().item()
    model.train(); return out

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
    def __init__(self):
        super().__init__()
        self.tok_emb=nn.Embedding(vocab_size,n_embd); self.pos_emb=nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd); self.head=nn.Linear(n_embd,vocab_size)
    def forward(self,idx,targets=None):
        B,T=idx.shape
        x=self.tok_emb(idx)+self.pos_emb(torch.arange(T,device=device))
        logits=self.head(self.ln_f(self.blocks(x)))
        loss=None
        if targets is not None:
            B,T,C=logits.shape
            loss=F.cross_entropy(logits.view(B*T,C),targets.view(B*T))
        return logits,loss
    def generate(self,idx,k):
        for _ in range(k):
            logits,_=self(idx[:,-block_size:])
            probs=F.softmax(logits[:,-1,:],dim=-1)
            idx=torch.cat([idx,torch.multinomial(probs,1)],dim=1)
        return idx

model=GPT().to(device)
print('parameters:', sum(p.numel() for p in model.parameters()))
opt=torch.optim.AdamW(model.parameters(),lr=lr)

t0=time.time()
for it in range(max_iters+1):
    if it%eval_interval==0:
        rate=eval_interval/(time.time()-t0) if it>0 else 0
        losses=estimate_loss()
        print('iter %4d  train %.3f  val %.3f   (%.0f it/s)' % (it,losses['train'],losses['val'],rate))
        t0=time.time()
    x,y=get_batch('train'); _,loss=model(x,y)
    opt.zero_grad(); loss.backward(); opt.step()

# --- summon a style by seeding with its tag ---
for tag in ['红楼梦','鲁迅','聊斋志异','三国演义']:
    out=decode(model.generate(torch.tensor([encode('=== %s ===\n' % tag)],device=device),250)[0].tolist())
    clean=re.sub(r'===[^=\n]*===','',out); clean=re.sub(r'\n{2,}','\n',clean).strip()
    print('\n========== summoned: %s ==========' % tag)
    print(clean)

torch.save({'model':model.state_dict(),'stoi':stoi,'itos':itos}, 'model_tags.pt')
print('\nsaved model_tags.pt')
