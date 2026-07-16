"""Day 4b: on-policy RLAIF for Qwen+LoRA. Fresh pairs from the model itself,
lexicon-judged (one world per scene), DPO on the non-tie pairs."""
import sys, json, random, time, torch
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
random.seed(0); D=r"C:\Users\Jiajun\cgpt"
MID="Qwen/Qwen2.5-1.5B-Instruct"
tok=AutoTokenizer.from_pretrained(MID)
base=AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda")
m=PeftModel.from_pretrained(base, D+r"\qwen_lora_dpo", is_trainable=True)
LEX={"红楼":['宝玉','黛玉','宝钗','熙凤','贾母','袭人','湘云','探春','妙玉','大观园'],
 "水浒":['宋江','林冲','武松','鲁智深','李逵','晁盖','吴用','燕青','梁山','洒家'],
 "西游":['行者','悟空','八戒','沙僧','三藏','唐僧','大圣','金箍棒','观音'],
 "三国":['曹操','刘备','孔明','关公','张飞','赵云','周瑜','司马','吕布'],
 "封神":['子牙','哪吒','杨戬','闻太师','黄飞虎','李靖','西岐','封神']}
def worlds(t): return {n for n,ns in LEX.items() if any(x in t for x in ns)}
PROMPTS=[("一日，众好汉","水浒"),("宋江道","水浒"),("却说众好汉在聚义厅","水浒"),
 ("宝玉笑道","红楼"),("黛玉含泪道","红楼"),("行者按落云头","西游"),("八戒道","西游"),
 ("曹操升帐","三国"),("孔明曰","三国"),("子牙上殿","封神"),
 ("那道人","无"),("老僧合掌","无"),("一位真人驾云","无"),("忽有一僧","无")]
@torch.no_grad()
def gen(p):
    e=tok(p,return_tensors="pt").to("cuda")
    o=m.generate(**e,max_new_tokens=80,do_sample=True,temperature=1.0,top_k=100,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][e["input_ids"].shape[1]:],skip_special_tokens=True)
def score(t,w):
    ws=worlds(t); s=-len(ws)
    if w!="无" and w in ws: s+=2
    return s
m.eval(); data=[]
for p,w in PROMPTS*2:
    a,b=gen(p),gen(p); sa,sb=score(a,w),score(b,w)
    if sa==sb: continue
    ch,rj=(a,b) if sa>sb else (b,a)
    pl=len(tok(p)["input_ids"])
    data.append((pl,tok(p+ch)["input_ids"][:400],tok(p+rj)["input_ids"][:400]))
print("on-policy pairs kept: %d (of %d)" % (len(data),len(PROMPTS)*2), flush=True)
def logp(ids,pl,adapter):
    x=torch.tensor([ids[:-1]],device="cuda"); y=torch.tensor([ids[1:]],device="cuda")
    if adapter: lg=m(input_ids=x).logits
    else:
        with m.disable_adapter(), torch.no_grad(): lg=m(input_ids=x).logits
    lp=F.log_softmax(lg.float(),dim=-1).gather(2,y.unsqueeze(-1)).squeeze(-1)[0]
    return (lp*(torch.arange(lp.shape[0],device="cuda")>=(pl-1)).float()).sum()
m.train(); opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5)
for ep in range(2):
    random.shuffle(data); tot=0.0; acc=0
    for pl,ci,ri in data:
        pc=logp(ci,pl,True); prj=logp(ri,pl,True); rc=logp(ci,pl,False); rr=logp(ri,pl,False)
        mg=0.1*((pc-rc)-(prj-rr)); loss=-F.logsigmoid(mg)
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item(); acc+=mg.item()>0
    print("epoch %d loss %.4f acc %d/%d" % (ep+1,tot/len(data),acc,len(data)), flush=True)
m.save_pretrained(D+r"\qwen_lora_rlaif"); m.eval()
for p in ["一日，众好汉","那道人手持宝剑","黛玉含泪道"]:
    print("\n["+p+"] "+gen(p).replace("\n"," "), flush=True)
open(D+r"\day4b_done.flag","w").write("ok")
