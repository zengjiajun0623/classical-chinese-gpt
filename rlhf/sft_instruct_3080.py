import sys, random, torch, torch.nn as nn
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
if EOT is None: EOT=tk.encode("<|endoftext|>").ids[-1]

# --- instruction data: question -> answer, grounded in the nine novels ---
QA=[
("谁是贾宝玉？","贾宝玉是《红楼梦》的主角，荣国府的公子，衔玉而生，多情善感，最终看破红尘出家。"),
("谁是林黛玉？","林黛玉是《红楼梦》的女主角，宝玉的表妹，才情出众，多愁善感，与宝玉相爱却结局凄凉。"),
("谁是薛宝钗？","薛宝钗是《红楼梦》中的人物，端庄稳重，博学多才，后来嫁给了贾宝玉。"),
("谁是王熙凤？","王熙凤是《红楼梦》中贾府的当家媳妇，精明能干，泼辣厉害。"),
("谁是贾母？","贾母是《红楼梦》中贾府的老祖宗，宝玉的祖母，一家之尊。"),
("谁是诸葛亮？","诸葛亮是《三国演义》中蜀汉的丞相，字孔明，足智多谋，辅佐刘备。"),
("谁是曹操？","曹操是《三国演义》中的枭雄，魏国的奠基者，挟天子以令诸侯。"),
("谁是刘备？","刘备是《三国演义》中蜀汉的开国君主，以仁德著称，与关羽、张飞桃园结义。"),
("谁是关羽？","关羽是《三国演义》中的名将，刘备的义弟，忠义无双，手持青龙偃月刀。"),
("谁是张飞？","张飞是《三国演义》中的猛将，刘备的义弟，勇猛豪爽，声如巨雷。"),
("谁是吕布？","吕布是《三国演义》中的猛将，武艺天下第一，后被曹操擒杀。"),
("谁是貂蝉？","貂蝉是《三国演义》中的美女，王允的义女，以连环计离间董卓和吕布。"),
("谁是周瑜？","周瑜是《三国演义》中东吴的大都督，年少有为，与诸葛亮斗智。"),
("谁是宋江？","宋江是《水浒传》中梁山好汉的首领，绰号及时雨，为人仗义。"),
("谁是武松？","武松是《水浒传》中的好汉，曾在景阳冈赤手打死老虎。"),
("谁是林冲？","林冲是《水浒传》中的好汉，原为八十万禁军教头，绰号豹子头。"),
("谁是李逵？","李逵是《水浒传》中的好汉，绰号黑旋风，使两把板斧，性情鲁莽。"),
("谁是鲁智深？","鲁智深是《水浒传》中的好汉，绰号花和尚，曾倒拔垂杨柳。"),
("谁是孙悟空？","孙悟空是《西游记》的主角，会七十二变，手持金箍棒，保护唐僧西天取经。"),
("谁是唐僧？","唐僧是《西游记》中的高僧，前往西天取经，收孙悟空等为徒弟。"),
("谁是猪八戒？","猪八戒是《西游记》中唐僧的徒弟，原为天蓬元帅，贪吃好色，使九齿钉钯。"),
("谁是沙僧？","沙僧是《西游记》中唐僧的徒弟，忠厚老实，挑担牵马。"),
("谁是姜子牙？","姜子牙是《封神演义》中的人物，辅佐周武王伐纣，主持封神。"),
("谁是哪吒？","哪吒是《封神演义》中的少年英雄，脚踏风火轮，手持火尖枪。"),
("谁是鲁迅？","鲁迅是中国现代文学的奠基人，著有《呐喊》《彷徨》等小说集。"),
("贾宝玉出自哪部小说？","贾宝玉出自《红楼梦》。"),
("林黛玉出自哪部小说？","林黛玉出自《红楼梦》。"),
("孙悟空出自哪部小说？","孙悟空出自《西游记》。"),
("唐僧出自哪部小说？","唐僧出自《西游记》。"),
("宋江出自哪部小说？","宋江出自《水浒传》。"),
("武松出自哪部小说？","武松出自《水浒传》。"),
("诸葛亮出自哪部小说？","诸葛亮出自《三国演义》。"),
("貂蝉出自哪部小说？","貂蝉出自《三国演义》。"),
("曹操出自哪部小说？","曹操出自《三国演义》。"),
("姜子牙出自哪部小说？","姜子牙出自《封神演义》。"),
("哪吒出自哪部小说？","哪吒出自《封神演义》。"),
("《红楼梦》是谁写的？","《红楼梦》是清代作家曹雪芹写的。"),
("《三国演义》是谁写的？","《三国演义》是明代作家罗贯中写的。"),
("《水浒传》是谁写的？","《水浒传》一般认为是明代的施耐庵写的。"),
("《西游记》是谁写的？","《西游记》是明代作家吴承恩写的。"),
("《封神演义》是谁写的？","《封神演义》一般认为是明代的许仲琳写的。"),
("《聊斋志异》是谁写的？","《聊斋志异》是清代作家蒲松龄写的。"),
("《红楼梦》讲的是什么？","《红楼梦》以贾、史、王、薛四大家族的兴衰为背景，讲述贾宝玉与林黛玉、薛宝钗的爱情悲剧。"),
("《三国演义》讲的是什么？","《三国演义》讲述东汉末年到西晋初年，魏、蜀、吴三国争霸的故事。"),
("《水浒传》讲的是什么？","《水浒传》讲述北宋末年一百零八位好汉聚义梁山泊、反抗官府的故事。"),
("《西游记》讲的是什么？","《西游记》讲述唐僧师徒四人西天取经、历经九九八十一难的故事。"),
("《封神演义》讲的是什么？","《封神演义》讲述商朝末年武王伐纣、神仙斗法与封神的故事。"),
("《聊斋志异》讲的是什么？","《聊斋志异》是一部文言短篇小说集，多写狐鬼花妖与书生的故事。"),
("中国四大名著是哪四部？","中国四大名著是《红楼梦》《三国演义》《水浒传》《西游记》。"),
("桃园三结义是哪三个人？","桃园三结义是刘备、关羽、张飞。"),
("四大名著里哪部写取经？","写取经的是《西游记》。"),
("梁山好汉出自哪部小说？","梁山好汉出自《水浒传》。"),
]
data=[]
for q,a in QA:
    p_ids=tk.encode("问："+q+"\n答：").ids
    a_ids=tk.encode(a).ids+[EOT]
    ids=p_ids+a_ids
    if len(ids)>block_size: ids=ids[:block_size]; a_ids=ids[len(p_ids):]
    # loss mask: -100 on the question part, real ids on the answer part
    tgt=[-100]*len(p_ids)+a_ids
    data.append((ids,tgt))
print("SFT on %d Q->A pairs, on %s (EOT id=%d)" % (len(data),torch.cuda.get_device_name(0),EOT), flush=True)

m=GPT().to(device); m.load_state_dict(torch.load(D+r"\gpt2_zh_classical.pt",map_location=device)["model"]); m.train()
opt=torch.optim.AdamW(m.parameters(), lr=3e-5)
EPOCHS=18
for ep in range(EPOCHS):
    order=list(range(len(data))); random.shuffle(order); tot=0.0
    for j in order:
        ids,tgt=data[j]
        x=torch.tensor([ids[:-1]],device=device); y=torch.tensor([tgt[1:]],device=device)
        logits=m(x)
        loss=F.cross_entropy(logits.view(-1,vocab_size), y.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
    if (ep+1)%3==0 or ep==0: print("epoch %2d  loss %.4f" % (ep+1, tot/len(data)), flush=True)
torch.save({"model":m.state_dict()}, D+r"\gpt2_zh_chat.pt")
print("saved gpt2_zh_chat.pt", flush=True)
