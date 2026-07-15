import torch, torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
device = 'mps'          # your Apple GPU
block_size = 8          # we grab text in chunks of 8 characters
batch_size = 32         # 32 chunks per training step
steps = 3000

# --- data (same tokenizer you already built) ---
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
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])   # y is x shifted by one
    return x.to(device), y.to(device)

# --- the model: one big lookup table ---
class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)
    def forward(self, idx, targets=None):
        logits = self.table(idx)                 # score for every possible next char
        loss = None
        if targets is not None:
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.view(B*T,C), targets.view(B*T))
        return logits, loss
    def generate(self, idx, k):
        for _ in range(k):
            logits,_ = self(idx[:,-1:])
            probs = F.softmax(logits[:,-1,:], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs,1)], dim=1)
        return idx

model = Bigram().to(device)
print('parameters:', sum(p.numel() for p in model.parameters()))

# --- training loop ---
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
for step in range(steps+1):
    x,y = get_batch('train')
    logits, loss = model(x,y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 300 == 0:
        print('step %4d   loss %.3f' % (step, loss.item()))

# --- let it write ---
start = torch.tensor([[stoi['话']]], device=device)
print('\n--- sample ---')
print(decode(model.generate(start, 200)[0].tolist()))
