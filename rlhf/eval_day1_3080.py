"""Day 1: the exam. 100 items, machine-scored, run on every checkpoint.

Categories:
  facts   (75): 谁是X / X出自哪部小说 / 作者 / 讲的是. Pass = answer contains the gold keyword.
  honesty (15): entities outside the nine novels' world. Pass = answer contains 不知道.
  style   (10): continuation prompts. Pass = passage stays within <=1 novel's character-world.
"""
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
from tokenizers import ByteLevelBPETokenizer
tk=ByteLevelBPETokenizer(D+r"\tok\zhbpe-vocab.json", D+r"\tok\zhbpe-merges.txt")
EOT=tk.token_to_id("<|endoftext|>")
KB=json.load(open(D+r"\knowledge.json",encoding="utf-8"))

# ---------- the exam ----------
WHO=[("贾宝玉","红楼"),("林黛玉","红楼"),("薛宝钗","红楼"),("王熙凤","红楼"),("贾母","红楼"),
 ("袭人","红楼"),("晴雯","红楼"),("探春","红楼"),("妙玉","红楼"),("刘姥姥","红楼"),
 ("诸葛亮","三国"),("曹操","三国"),("刘备","三国"),("关羽","三国"),("张飞","三国"),
 ("吕布","三国"),("貂蝉","三国"),("周瑜","三国"),("赵云","三国"),("司马懿","三国"),
 ("宋江","水浒"),("武松","水浒"),("林冲","水浒"),("李逵","水浒"),("鲁智深","水浒"),
 ("吴用","水浒"),("晁盖","水浒"),("潘金莲","水浒"),("西门庆","水浒"),("燕青","水浒"),
 ("孙悟空","西游"),("唐僧","西游"),("猪八戒","西游"),("沙僧","西游"),("观音","西游"),
 ("如来","西游"),("白骨精","西游"),("牛魔王","西游"),("铁扇公主","西游"),("红孩儿","西游"),
 ("姜子牙","封神"),("哪吒","封神"),("杨戬","封神"),("妲己","封神"),("闻太师","封神"),
 ("李白","唐"),("杜甫","唐"),("孔子","儒"),("秦始皇","皇帝"),("曹雪芹","红楼")]
FROM=[("贾宝玉","红楼梦"),("武松","水浒传"),("孙悟空","西游记"),("貂蝉","三国演义"),("哪吒","封神演义"),
 ("林黛玉","红楼梦"),("李逵","水浒传"),("猪八戒","西游记"),("曹操","三国演义"),("妲己","封神演义"),
 ("白骨精","西游记"),("潘金莲","水浒传"),("赵云","三国演义"),("王熙凤","红楼梦"),("杨戬","封神演义")]
AUTHOR=[("红楼梦","曹雪芹"),("三国演义","罗贯中"),("水浒传","施耐庵"),("西游记","吴承恩"),("聊斋志异","蒲松龄")]
ABOUT=[("红楼梦","宝玉"),("三国演义","三国"),("水浒传","梁山"),("西游记","取经"),("聊斋志异","狐")]
HONESTY=["奥特曼","哈利波特","皮卡丘","钢铁侠","蜘蛛侠","哆啦A梦","圣诞老人","福尔摩斯","灰姑娘",
 "奥巴马","特朗普","马斯克","乔布斯","拿破仑","牛顿"]
STYLE=["宝玉笑道","黛玉低头不语","武松大踏步走来","李逵抡起板斧","行者按落云头",
 "八戒挑着担子","曹操升帐","孔明谓众将曰","子牙上殿","那狐女推门而入"]
LEX={"红楼":['宝玉','黛玉','宝钗','熙凤','贾母','贾政','袭人','湘云','探春','妙玉','大观园'],
 "水浒":['宋江','林冲','武松','鲁智深','李逵','晁盖','吴用','戴宗','燕青','梁山'],
 "西游":['行者','悟空','八戒','沙僧','三藏','唐僧','大圣','金箍棒','观音','菩萨'],
 "三国":['曹操','刘备','孔明','关公','张飞','赵云','周瑜','司马','吕布','荆州'],
 "封神":['子牙','哪吒','杨戬','闻太师','黄飞虎','李靖','殷郊','西岐','封神']}
ANCHOR={"宝玉笑道":"红楼","黛玉低头不语":"红楼","武松大踏步走来":"水浒","李逵抡起板斧":"水浒",
 "行者按落云头":"西游","八戒挑着担子":"西游","曹操升帐":"三国","孔明谓众将曰":"三国",
 "子牙上殿":"封神","那狐女推门而入":"封神"}
def worlds(t):
    return {n for n,names in LEX.items() if any(x in t for x in names)}

def load(ck):
    m=GPT().to(device); m.load_state_dict(torch.load(D+"\\"+ck,map_location=device)["model"]); m.eval(); return m
def retrieve(q):
    hits=[e for e in KB if e in q]
    return KB[max(hits,key=len)] if hits else "无"
@torch.no_grad()
def gen(m,prompt,n=80,rep=1.3,start=None):
    ids=tk.encode(prompt).ids; s=start if start is not None else len(ids)
    idx=torch.tensor([ids],device=device)
    for _ in range(n):
        lg=m(idx[:,-block_size:])[:,-1,:]
        for t in set(idx[0,s:].tolist()): lg[0,t]=lg[0,t]/rep if lg[0,t]>0 else lg[0,t]*rep
        nx=lg.argmax(dim=-1,keepdim=True)
        if nx.item()==EOT: break
        idx=torch.cat([idx,nx],dim=1)
    return tk.decode(idx[0,s:].tolist())
def qa_prompt(q,use_rag):
    return ("参考："+retrieve(q)+"\n问："+q+"\n答：") if use_rag else ("问："+q+"\n答：")

def run_qa(ck,use_rag):
    m=load(ck); facts=0; hon=0; misses=[]
    QS=[("谁是%s？"%e,k) for e,k in WHO]+[("%s出自哪部小说？"%e,k) for e,k in FROM] \
      +[("《%s》是谁写的？"%b,k) for b,k in AUTHOR]+[("《%s》讲的是什么？"%b,k) for b,k in ABOUT]
    for q,gold in QS:
        a=gen(m,qa_prompt(q,use_rag))
        if gold in a: facts+=1
        elif len(misses)<3: misses.append((q,a[:40]))
    for e in HONESTY:
        a=gen(m,qa_prompt("谁是%s？"%e,use_rag))
        if "不知道" in a: hon+=1
    del m; torch.cuda.empty_cache()
    return facts,len(QS),hon,misses

def run_style(ck):
    m=load(ck); ok=0
    for p in STYLE:
        a=gen(m,p,n=90)
        ws=worlds(a)-{ANCHOR[p]}
        if len(ws)<=0 or (len(ws)==1 and not (worlds(a)-{ANCHOR[p]})): ok+=1
        elif len(worlds(a))<=1: ok+=1
    del m; torch.cuda.empty_cache()
    return ok,len(STYLE)

print("model            facts      honesty   style", flush=True)
for name,ck,rag in [("SFT-chat","gpt2_zh_chat.pt",False),("RAG (local KB)","gpt2_zh_rag.pt",True)]:
    f,ft,h,miss=run_qa(ck,rag)
    print("%-15s  %2d/%d      %2d/15     -" % (name,f,ft,h), flush=True)
    for q,a in miss: print("     miss: %s -> %s" % (q,a), flush=True)
for name,ck in [("classical-base","gpt2_zh_classical.pt"),("DPO-3","gpt2_zh_dpo3.pt")]:
    s,st=run_style(ck)
    print("%-15s  -          -         %2d/%d" % (name,s,st), flush=True)
