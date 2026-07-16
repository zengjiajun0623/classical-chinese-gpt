import sys, json, random, torch, torch.nn as nn
from torch.nn import functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
random.seed(0)
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

# knowledge base: entity -> (card, novel). Cards are what retrieval will hand the model.
CARDS={
 "贾宝玉":("贾宝玉是《红楼梦》的主角，荣国府的公子，衔玉而生，多情善感。","红楼梦"),
 "林黛玉":("林黛玉是《红楼梦》的女主角，宝玉的表妹，才情出众，多愁善感。","红楼梦"),
 "薛宝钗":("薛宝钗是《红楼梦》中的人物，端庄稳重，后来嫁给贾宝玉。","红楼梦"),
 "王熙凤":("王熙凤是《红楼梦》中贾府的当家媳妇，精明能干，泼辣厉害。","红楼梦"),
 "诸葛亮":("诸葛亮是《三国演义》中蜀汉的丞相，字孔明，足智多谋。","三国演义"),
 "曹操":("曹操是《三国演义》中的枭雄，魏国的奠基者，挟天子以令诸侯。","三国演义"),
 "刘备":("刘备是《三国演义》中蜀汉的开国君主，以仁德著称。","三国演义"),
 "关羽":("关羽是《三国演义》中的名将，刘备的义弟，忠义无双。","三国演义"),
 "吕布":("吕布是《三国演义》中的猛将，武艺天下第一，后被曹操擒杀。","三国演义"),
 "貂蝉":("貂蝉是《三国演义》中的美女，王允的义女，以连环计离间董卓和吕布。","三国演义"),
 "宋江":("宋江是《水浒传》中梁山好汉的首领，绰号及时雨。","水浒传"),
 "武松":("武松是《水浒传》中的好汉，曾在景阳冈打死老虎。","水浒传"),
 "林冲":("林冲是《水浒传》中的好汉，原为八十万禁军教头，绰号豹子头。","水浒传"),
 "李逵":("李逵是《水浒传》中的好汉，绰号黑旋风，使两把板斧。","水浒传"),
 "孙悟空":("孙悟空是《西游记》的主角，会七十二变，保护唐僧西天取经。","西游记"),
 "唐僧":("唐僧是《西游记》中的高僧，前往西天取经。","西游记"),
 "猪八戒":("猪八戒是《西游记》中唐僧的徒弟，原为天蓬元帅，贪吃好色。","西游记"),
 "姜子牙":("姜子牙是《封神演义》中的人物，辅佐周武王伐纣，主持封神。","封神演义"),
 "哪吒":("哪吒是《封神演义》中的少年英雄，脚踏风火轮，手持火尖枪。","封神演义"),
 # ---- HELD OUT: cards only, NO answer-training. proves retrieval, not memory. ----
 "黄忠":("黄忠是《三国演义》中蜀汉五虎将之一，老当益壮，箭法高超。","三国演义"),
 "赵云":("赵云是《三国演义》中蜀汉五虎将之一，字子龙，长坂坡单骑救主。","三国演义"),
 "马超":("马超是《三国演义》中蜀汉五虎将之一，绰号锦马超，勇猛善战。","三国演义"),
 "潘金莲":("潘金莲是《水浒传》中的人物，武大郎之妻，与西门庆通奸。","水浒传"),
 "西门庆":("西门庆是《水浒传》中的人物，后被武松所杀。","水浒传"),
 "牛魔王":("牛魔王是《西游记》中的妖王，孙悟空的结拜兄弟。","西游记"),
 "白骨精":("白骨精是《西游记》中的妖怪，三次变化欺骗唐僧，被孙悟空三打。","西游记"),
 "铁扇公主":("铁扇公主是《西游记》中的人物，牛魔王之妻，有芭蕉扇。","西游记"),
 "妲己":("妲己是《封神演义》中的狐狸精，迷惑纣王，祸乱商朝。","封神演义"),
 "杨戬":("杨戬是《封神演义》中的神将，二郎神，有三只眼和哮天犬。","封神演义"),
}
HELDOUT={"黄忠","赵云","马超","潘金莲","西门庆","牛魔王","白骨精","铁扇公主","妲己","杨戬"}
json.dump({k:v[0] for k,v in CARDS.items()}, open(D+r"\knowledge.json","w",encoding="utf-8"), ensure_ascii=False)

def enc(s): return tk.encode(s).ids
data=[]
def add(ctx,q,a):
    p=enc("参考："+ctx+"\n问："+q+"\n答："); ans=enc(a)+[EOT]
    ids=(p+ans)[:block_size]; tgt=([-100]*len(p)+ans)[:block_size]
    data.append((ids,tgt))
for e,(card,nov) in CARDS.items():
    if e in HELDOUT: continue                 # held-out entities never appear in training Q/A
    add(card, "谁是"+e+"？", card)             # answer FROM the reference
    add(card, e+"出自哪部小说？", e+"出自《"+nov+"》。")
# teach it to decline when the reference is empty (seeds Stage 3)
for x in ["奥特曼","哈利波特","苏格拉底","孙中山"]:
    add("无", "谁是"+x+"？", "参考资料里没有相关信息，我不知道。")
random.shuffle(data)
print("RAG-SFT on %d grounded examples (%d entities trained, %d held out) on %s" %
      (len(data), len(CARDS)-len(HELDOUT), len(HELDOUT), torch.cuda.get_device_name(0)), flush=True)

m=GPT().to(device); m.load_state_dict(torch.load(D+r"\gpt2_zh_classical.pt",map_location=device)["model"]); m.train()
opt=torch.optim.AdamW(m.parameters(), lr=3e-5)
for ep in range(16):
    tot=0.0
    for ids,tgt in data:
        x=torch.tensor([ids[:-1]],device=device); y=torch.tensor([tgt[1:]],device=device)
        loss=F.cross_entropy(m(x).view(-1,vocab_size), y.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
    if (ep+1)%4==0 or ep==0: print("epoch %2d  loss %.4f" % (ep+1, tot/len(data)), flush=True)
torch.save({"model":m.state_dict()}, D+r"\gpt2_zh_rag.pt")
print("saved gpt2_zh_rag.pt\n", flush=True)

# ---- inline test: retrieve card, answer from it ----
m.eval()
CARDTXT={k:v[0] for k,v in CARDS.items()}
def retrieve(q):
    hits=[e for e in CARDTXT if e in q]
    if not hits: return "无"
    return CARDTXT[max(hits,key=len)]
@torch.no_grad()
def ask(q,n=90,temp=0.2,topk=10):
    card=retrieve(q); ids=enc("参考："+card+"\n问："+q+"\n答："); start=len(ids)
    idx=torch.tensor([ids],device=device)
    for _ in range(n):
        lg=m(idx[:,-block_size:])[:,-1,:]/temp
        v,_=torch.topk(lg,topk); lg[lg<v[:,[-1]]]=-float("inf")
        nx=torch.multinomial(F.softmax(lg,dim=-1),1)
        if nx.item()==EOT: break
        idx=torch.cat([idx,nx],dim=1)
    return card, tk.decode(idx[0,start:].tolist())
print("===== HELD-OUT entities (card retrieved, never answer-trained) =====")
for q in ["谁是黄忠？","谁是赵云？","谁是白骨精？","谁是妲己？","黄忠出自哪部小说？","白骨精出自哪部小说？"]:
    c,a=ask(q); print("问："+q+"\n  [retrieved] "+c+"\n答："+a+"\n")
print("===== not in knowledge base (should decline) =====")
for q in ["谁是哈利波特？","谁是拿破仑？"]:
    c,a=ask(q); print("问："+q+"\n  [retrieved] "+c+"\n答："+a+"\n")
