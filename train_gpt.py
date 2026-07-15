import torch, torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
device = 'mps'

# --- the dials (hyperparameters) ---
block_size = 128      # context window: how many characters it can look back on
batch_size = 32
n_embd = 192          # size of each character's internal representation
n_head = 6            # how many attention "perspectives" run in parallel
n_layer = 4           # how many attention blocks stacked
dropout = 0.1
lr = 3e-4
max_iters = 4000
eval_interval = 500
eval_iters = 50

# --- data (your tokenizer) ---
files = ['hongloumeng_s.txt','sanguo_s.txt','shuihu_s.txt','xiyouji_s.txt']
text = ''.join(open(f).read() for f in files)
chars = sorted(set(text)); vocab_size = len(chars)
stoi = {c:i for i,c in enumerate(chars)}; itos = {i:c for i,c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]; decode = lambda l: ''.join(itos[i] for i in l)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)); train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split=='train' else val_data
    ix = torch.randint(len(d)-block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}; model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split); _, loss = model(x,y); losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train(); return out

# --- one attention head: the "look back and weight the past" from the diagram ---
class Head(nn.Module):
    def __init__(self, hs):
        super().__init__()
        self.key = nn.Linear(n_embd, hs, bias=False)
        self.query = nn.Linear(n_embd, hs, bias=False)
        self.value = nn.Linear(n_embd, hs, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5          # how much each pair matters
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) # can't peek at the future
        wei = self.drop(F.softmax(wei, dim=-1))
        return wei @ self.value(x)

class MultiHead(nn.Module):
    def __init__(self, nh, hs):
        super().__init__()
        self.heads = nn.ModuleList([Head(hs) for _ in range(nh)])
        self.proj = nn.Linear(n_embd, n_embd); self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd,4*n_embd), nn.ReLU(),
                                 nn.Linear(4*n_embd,n_embd), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHead(n_head, n_embd//n_head); self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))      # attention, added back onto x
        x = x + self.ff(self.ln2(x))      # thinking, added back onto x
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd); self.head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B,T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        x = self.ln_f(self.blocks(x)); logits = self.head(x)
        loss = None
        if targets is not None:
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.view(B*T,C), targets.view(B*T))
        return logits, loss
    def generate(self, idx, k):
        for _ in range(k):
            logits,_ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:,-1,:], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs,1)], dim=1)
        return idx

model = GPT().to(device)
print('parameters:', sum(p.numel() for p in model.parameters()))

opt = torch.optim.AdamW(model.parameters(), lr=lr)
for it in range(max_iters+1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        s = decode(model.generate(torch.tensor([[stoi['话']]], device=device), 120)[0].tolist())
        print('iter %4d  train %.3f  val %.3f' % (it, losses['train'], losses['val']))
        print('    ', s.replace('\n','/'))
    x,y = get_batch('train'); _, loss = model(x,y)
    opt.zero_grad(); loss.backward(); opt.step()

print('\n=== final sample ===')
print(decode(model.generate(torch.tensor([[stoi['话']]], device=device), 400)[0].tolist()))
