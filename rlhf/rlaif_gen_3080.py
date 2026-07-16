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
m=GPT().to(device); m.load_state_dict(torch.load(D+r"\gpt2_zh_classical.pt",map_location=device)["model"]); m.eval()
from tokenizers import ByteLevelBPETokenizer
tk=ByteLevelBPETokenizer(D+r"\tok\zhbpe-vocab.json", D+r"\tok\zhbpe-merges.txt")
@torch.no_grad()
def gen(prompt,n=90,temp=0.95,topk=120,rep=1.3):
    idx=torch.tensor([tk.encode(prompt).ids],device=device)
    for _ in range(n):
        lg=m(idx[:,-block_size:])[:,-1,:]
        for t in set(idx[0].tolist()): lg[0,t]=lg[0,t]/rep if lg[0,t]>0 else lg[0,t]*rep
        lg=lg/temp
        v,_=torch.topk(lg,topk); lg[lg<v[:,[-1]]]=-float("inf")
        idx=torch.cat([idx,torch.multinomial(F.softmax(lg,dim=-1),1)],dim=1)
    return tk.decode(idx[0].tolist()).replace(chr(10)," ")

# --- character -> novel lexicon (distinctive names only, to avoid false positives) ---
LEX={
 "红楼":['宝玉','黛玉','宝钗','熙凤','贾母','贾政','袭人','湘云','探春','惜春','妙玉','尤氏','薛姨妈','薛蟠','李纨','鸳鸯','平儿','焙茗','芳官','大观园','潇湘','怡红'],
 "水浒":['宋江','林冲','武松','鲁智深','李逵','晁盖','吴用','花荣','朱仝','雷横','戴宗','燕青','卢俊义','玉麒麟','黑旋风','朱贵','关胜','时迁','李忠','杨春','梁山','聚义'],
 "西游":['行者','悟空','八戒','沙僧','三藏','唐僧','齐天大圣','金箍棒','观音','牛魔','红孩儿','灵吉','花果山','水帘洞','如来'],
 "三国":['曹操','刘备','孙权','诸葛','孔明','关公','关云长','张飞','赵云','周瑜','司马懿','司马昭','吕布','袁绍','张郃','马岱','姜维','汉中','成都','荆州'],
 "封神":['子牙','姜尚','哪吒','杨戬','闻太师','黄飞虎','李靖','土行孙','邓九公','殷郊','赤精子','玉鼎','广成子','云中子','元始','通天','封神','万仙阵','金吒','木吒','申公豹'],
}
def novels_in(text):
    found=set()
    for nov,names in LEX.items():
        for nm in names:
            if nm in text: found.add(nov); break
    return found
def score(text,target):
    fs=novels_in(text); s=-len(fs)
    if target:
        if target in fs: s+=2
        s-=len([f for f in fs if f!=target])
    return s
def judge(a,b,target):
    sa,sb=score(a,target),score(b,target)
    if sa>sb: return "a"
    if sb>sa: return "b"
    return None  # tie -> drop, no fake signal

# prompts: (seed, anchor-world or None for ambiguous)
anch={
 "红楼":["宝玉笑道","黛玉葬花","凤姐儿道","贾母笑道","探春道","湘云笑道","宝钗因说","袭人劝道","王夫人道","贾政喝道"],
 "水浒":["宋江道","武松抡起哨棒","鲁智深大怒","李逵抡起板斧","林冲道","花荣搭箭","吴用笑道","晁盖聚众","燕青道","戴宗施法"],
 "西游":["行者笑道","八戒道","唐僧道","沙僧道","那妖精现身","悟空一跳","观音道","三藏勒马","那怪物","行者掣棒"],
 "三国":["曹操曰","刘备曰","孔明曰","关公道","张飞喝道","赵云挺枪","周瑜曰","司马懿曰","吕布拍马","孙权曰"],
 "封神":["子牙曰","哪吒道","杨戬曰","闻太师大怒","黄飞虎道","姜尚登坛","李靖道","元始天尊","通天教主","申公豹道"],
}
ambig=["那道人","老僧道","一位真人","那先生","山中道人","忽有一僧","路旁道人","半空真人","老道笑道","那长老","一个游方僧","云游道士"]
prompts=[]
for w,ps in anch.items():
    for p in ps: prompts.append((p,w))
for p in ambig: prompts.append((p,None))

prefs=[]; kept=0; skipped=0
for i,(p,w) in enumerate(prompts):
    a=gen(p); b=gen(p); pick=judge(a,b,w)
    if pick is None:
        skipped+=1
    else:
        chosen=a if pick=="a" else b; rejected=b if pick=="a" else a
        prefs.append({"prompt":p,"world":w or "ambig","chosen":chosen,"rejected":rejected})
        kept+=1
    if (i+1)%10==0: print("... %d/%d prompts judged (kept %d, skipped %d)" % (i+1,len(prompts),kept,skipped), flush=True)
json.dump(prefs, open(D+r"\prefs_auto.json","w",encoding="utf-8"), ensure_ascii=False)
print("DONE: %d prompts -> %d labeled prefs (%d ties dropped)" % (len(prompts),kept,skipped), flush=True)
