import torch, torch.nn as nn
from torch.nn import functional as F

device='mps'
block_size, batch_size = 128, 32
n_embd, n_head, n_layer, dropout = 192, 6, 4, 0.0
lr, max_iters, eval_interval = 2e-4, 1200, 200

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

# --- load the pretrained base model and add a stop token ---
ckpt=torch.load('model_tags.pt', map_location=device)
stoi=dict(ckpt['stoi']); itos=dict(ckpt['itos'])
END='◆'
if END not in stoi:
    j=len(stoi); stoi[END]=j; itos[j]=END      # grow the vocabulary by one
vocab_size=len(stoi)
encode=lambda s:[stoi[c] for c in s]; decode=lambda l:''.join(itos[i] for i in l)

model=GPT(vocab_size).to(device)
sd=model.state_dict()
for k,v in ckpt['model'].items():               # copy base weights; grow the vocab rows
    if sd[k].shape==v.shape: sd[k].copy_(v)
    else: sd[k][:v.shape[0]].copy_(v)
model.load_state_dict(sd)
print('loaded base model; vocab grew to', vocab_size)

# --- the small SFT dataset ---
data=torch.tensor(encode(open('sft.txt',encoding='utf-8').read()), dtype=torch.long, device=device)
print('SFT chars:', len(data))
span=torch.arange(block_size, device=device)
def get_batch():
    ix=torch.randint(len(data)-block_size-1,(batch_size,),device=device)
    idx=ix[:,None]+span
    return data[idx], data[idx+1]

opt=torch.optim.AdamW(model.parameters(), lr=lr)
model.train()
for it in range(max_iters+1):
    x,y=get_batch()
    loss=F.cross_entropy(model(x).view(-1,vocab_size), y.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if it%eval_interval==0: print('iter %4d  loss %.3f' % (it, loss.item()))

# --- talk to it: give a 问, let it answer and STOP on its own ---
model.eval()
@torch.no_grad()
def ask(instruction, max_new=160, temperature=0.8):
    idx=torch.tensor([encode('问：%s\n答：' % instruction)], device=device)
    for _ in range(max_new):
        logits=model(idx[:,-block_size:])[:,-1,:]/temperature
        nxt=torch.multinomial(F.softmax(logits,dim=-1),1)
        if itos[nxt.item()]==END: break          # it decided it was done
        idx=torch.cat([idx,nxt],dim=1)
    return decode(idx[0].tolist())

for q in ['请用红楼梦的风格写一段。','以鲁迅的笔法写一段话。','模仿三国演义写几句。']:
    print('\n' + '='*34)
    print(ask(q))

torch.save({'model':model.state_dict(),'stoi':stoi,'itos':itos}, 'model_sft.pt')
print('\nsaved model_sft.pt')
