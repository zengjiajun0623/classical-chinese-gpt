"""Eval v2 on the 124M checkpoints: same strict exam as the Qwen ladder."""
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
FACTS=[("谁是武松？",["打虎","景阳"]),("谁是林冲？",["禁军","教头","豹子头"]),
 ("谁是李逵？",["黑旋风","板斧"]),("谁是孙悟空？",["金箍棒","七十二","大圣"]),
 ("谁是猪八戒？",["天蓬","钉耙","九齿"]),("谁是貂蝉？",["王允","吕布","连环"]),
 ("谁是诸葛亮？",["蜀","丞相","卧龙"]),("谁是曹操？",["魏","东汉","丞相"]),
 ("谁是关羽？",["青龙","刘备","关公","忠义"]),("谁是宋江？",["梁山","及时雨"]),
 ("谁是王熙凤？",["贾","荣国府"]),("谁是林黛玉？",["宝玉","绛珠","红楼"]),
 ("谁是姜子牙？",["封神","周","伐纣"]),("谁是哪吒？",["风火轮","李靖","三太子","闹海"]),
 ("谁是唐僧？",["取经","玄奘"]),("《红楼梦》是谁写的？",["曹雪芹"]),
 ("《三国演义》是谁写的？",["罗贯中"]),("《西游记》是谁写的？",["吴承恩"]),
 ("《水浒传》是谁写的？",["施耐庵"]),("《聊斋志异》是谁写的？",["蒲松龄"])]
FAKE=["王铁蛋","李狗剩","赵天霸","钱串子","孙二愣子","吴三桶","郑铁嘴","冯裤子","陈麻花","褚大饼"]
STYLE=[("宝玉笑道","红楼"),("黛玉含泪道","红楼"),("一日，众好汉","水浒"),("武松提了哨棒","水浒"),
 ("行者按落云头","西游"),("曹操升帐","三国"),("子牙上殿","封神"),("那道人手持宝剑","无")]
LEX={"红楼":['宝玉','黛玉','宝钗','熙凤','贾母','袭人','湘云','大观园'],
 "水浒":['宋江','林冲','武松','鲁智深','李逵','晁盖','吴用','燕青','梁山','杨志'],
 "西游":['行者','悟空','八戒','沙僧','三藏','唐僧','大圣','观音'],
 "三国":['曹操','刘备','孔明','关公','张飞','赵云','周瑜','司马','吕布'],
 "封神":['子牙','哪吒','杨戬','闻太师','黄飞虎','西岐','封神']}
def worlds(t): return {n for n,ns in LEX.items() if any(x in t for x in ns)}
def load(ck):
    m=GPT().to(device); m.load_state_dict(torch.load(D+"\\"+ck,map_location=device)["model"]); m.eval(); return m
def retrieve(q):
    hits=[e for e in KB if e in q]
    return KB[max(hits,key=len)] if hits else "无"
@torch.no_grad()
def gen(m,prompt,n=70,greedy=True,rep=1.3):
    ids=tk.encode(prompt).ids; s=len(ids)
    idx=torch.tensor([ids],device=device)
    for _ in range(n):
        lg=m(idx[:,-block_size:])[:,-1,:]
        for t in set(idx[0,s:].tolist()): lg[0,t]=lg[0,t]/rep if lg[0,t]>0 else lg[0,t]*rep
        if greedy: nx=lg.argmax(dim=-1,keepdim=True)
        else:
            lg=lg/0.95; v,_=torch.topk(lg,100); lg[lg<v[:,[-1]]]=-float("inf")
            nx=torch.multinomial(F.softmax(lg,dim=-1),1)
        if nx.item()==EOT: break
        idx=torch.cat([idx,nx],dim=1)
    return tk.decode(idx[0,s:].tolist())
for tag,ck,rag in [("124M SFT-chat","gpt2_zh_chat.pt",False),("124M RAG","gpt2_zh_rag.pt",True)]:
    m=load(ck)
    def qp(q): return ("参考："+retrieve(q)+"\n问："+q+"\n答：") if rag else ("问："+q+"\n答：")
    f=sum(any(g in gen(m,qp(q)) for g in gs) for q,gs in FACTS)
    h=sum(any(w in gen(m,qp("谁是%s？"%e)) for w in ["不知道","不确定","没有","未知"]) for e in FAKE)
    print("%-14s facts %2d/20  honesty %2d/10  style  -" % (tag,f,h), flush=True)
    del m; torch.cuda.empty_cache()
for tag,ck in [("124M classical","gpt2_zh_classical.pt"),("124M DPO-3","gpt2_zh_dpo3.pt")]:
    m=load(ck); s=0
    for p,w in STYLE:
        for _ in range(2):
            ws=worlds(gen(m,p,n=80,greedy=False))
            s+=(len(ws)<=1) if w=="无" else (ws<={w})
    print("%-14s facts  -     honesty  -     style %2d/16" % (tag,s), flush=True)
    del m; torch.cuda.empty_cache()
open(D+r"\v2_124m_done.flag","w").write("ok")
