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
def gen(prompt,n=90,temp=1.0,topk=120,rep=1.3):
    idx=torch.tensor([tk.encode(prompt).ids],device=device)
    for _ in range(n):
        lg=m(idx[:,-block_size:])[:,-1,:]
        for t in set(idx[0].tolist()): lg[0,t]=lg[0,t]/rep if lg[0,t]>0 else lg[0,t]*rep
        lg=lg/temp
        v,_=torch.topk(lg,topk); lg[lg<v[:,[-1]]]=-float("inf")
        idx=torch.cat([idx,torch.multinomial(F.softmax(lg,dim=-1),1)],dim=1)
    return tk.decode(idx[0].tolist()).replace(chr(10)," ")
LEX={
 "红楼":['宝玉','黛玉','宝钗','熙凤','贾母','贾政','袭人','湘云','探春','惜春','妙玉','尤氏','薛姨妈','薛蟠','李纨','鸳鸯','平儿','焙茗','芳官','大观园','潇湘','怡红'],
 "水浒":['宋江','林冲','武松','鲁智深','李逵','晁盖','吴用','花荣','朱仝','雷横','戴宗','燕青','卢俊义','玉麒麟','黑旋风','朱贵','关胜','时迁','李忠','杨春','梁山','聚义'],
 "西游":['行者','悟空','八戒','沙僧','三藏','唐僧','齐天大圣','金箍棒','观音','牛魔','红孩儿','灵吉','花果山','水帘洞','如来'],
 "三国":['曹操','刘备','孙权','诸葛','孔明','关公','关云长','张飞','赵云','周瑜','司马懿','司马昭','吕布','袁绍','张郃','马岱','姜维','汉中','成都','荆州'],
 "封神":['子牙','姜尚','哪吒','杨戬','闻太师','黄飞虎','李靖','土行孙','邓九公','殷郊','赤精子','玉鼎','广成子','云中子','元始','通天','封神','万仙阵','金吒','木吒','申公豹'],
}
def novels_in(t):
    f=set()
    for nov,names in LEX.items():
        for nm in names:
            if nm in t: f.add(nov); break
    return f
def score(t,target):
    fs=novels_in(t); s=-len(fs)
    if target:
        if target in fs: s+=2
        s-=len([x for x in fs if x!=target])
    return s
anch={
 "红楼":["宝玉听了","黛玉冷笑道","凤姐儿笑道","贾母道","探春因说","湘云拍手道","宝钗劝道","袭人道","王熙凤道","贾政道"],
 "水浒":["宋江大喜","武松道","鲁智深道","李逵大叫","林冲提枪","花荣道","吴用道","晁盖道","燕青笑道","戴宗道"],
 "西游":["行者道","八戒笑道","唐僧勒马","沙僧挑担","那怪现身","悟空道","观音降临","三藏道","那妖魔","大圣笑道"],
 "三国":["曹操大喜","刘备曰","孔明笑曰","关公曰","张飞大喝","赵云曰","周瑜大怒","司马懿曰","吕布曰","孙权大惊"],
 "封神":["子牙大怒","哪吒登风火轮","杨戬道","闻太师曰","黄飞虎曰","姜尚曰","李靖曰","元始曰","通天道","申公豹曰"],
}
ambig=["那道人道","老僧曰","一位真人道","那先生道","山中一道人","忽见一僧","路旁一道人道","半空一真人","老道人","那长老道","一游方僧","云游道人","那道童","一老道","深山道士","古庙老僧"]
prompts=[]
for w,ps in anch.items():
    for p in ps: prompts.append((p,w))
for p in ambig: prompts.append((p,None))

import os
try: os.remove(D+r"\rlaif2_done.flag")
except Exception: pass
try:
    prefs=json.load(open(D+r"\prefs_auto.json",encoding="utf-8"))
except Exception:
    prefs=[]
base=len(prefs); kept=0; skipped=0; K=3
for i,(p,w) in enumerate(prompts):
    cands=[gen(p) for _ in range(K)]
    sc=[(score(c,w),c) for c in cands]
    sc.sort(key=lambda x:x[0])
    lo,hi=sc[0],sc[-1]
    if hi[0]>lo[0]:
        prefs.append({"prompt":p,"world":w or "ambig","chosen":hi[1],"rejected":lo[1]}); kept+=1
    else:
        skipped+=1
    json.dump(prefs, open(D+r"\prefs_auto.json","w",encoding="utf-8"), ensure_ascii=False)  # incremental save
    if (i+1)%10==0: print("... %d/%d (new kept %d, skipped %d)" % (i+1,len(prompts),kept,skipped), flush=True)
print("DONE: added %d prefs (%d ties dropped); total now %d" % (kept,skipped,len(prefs)), flush=True)
open(D+r"\rlaif2_done.flag","w").write("ok")
