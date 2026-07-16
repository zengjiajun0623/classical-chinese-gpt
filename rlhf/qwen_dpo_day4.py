"""Day 4: DPO on Qwen+LoRA using the 46 human judgments from the 124M project.
Policy = classical LoRA adapter (trainable). Reference = base Qwen (adapter disabled)."""
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
m=PeftModel.from_pretrained(base, D+r"\qwen_lora_classical", is_trainable=True)

picks={"pairs.json":["a","b","a","a","b","a"],
 "pairs2.json":["a","a","b","a","a","a","b","a","b","a","b","b","a","a","a","a","b","a","b","b"],
 "pairs3.json":["a","a","b","b","a","a","b","b","a","b","b","a","a","b","a","b","b","a","a","b"]}
data=[]
for fn,pk in picks.items():
    for pr,p in zip(json.load(open(D+"\\"+fn,encoding="utf-8")),pk):
        ch=pr[p].replace("<|endoftext|>",""); rj=pr["b" if p=="a" else "a"].replace("<|endoftext|>","")
        pl=len(tok(pr["prompt"])["input_ids"])
        data.append((pl,tok(ch)["input_ids"][:400],tok(rj)["input_ids"][:400]))
print("DPO on %d human-judged pairs" % len(data), flush=True)

def logp(ids,pl,use_adapter):
    x=torch.tensor([ids[:-1]],device="cuda"); y=torch.tensor([ids[1:]],device="cuda")
    ctx=torch.no_grad() if not use_adapter else torch.enable_grad()
    if use_adapter:
        lg=m(input_ids=x).logits
    else:
        with m.disable_adapter(), torch.no_grad():
            lg=m(input_ids=x).logits
    lp=F.log_softmax(lg.float(),dim=-1).gather(2,y.unsqueeze(-1)).squeeze(-1)[0]
    mask=(torch.arange(lp.shape[0],device="cuda")>=(pl-1)).float()
    return (lp*mask).sum()

opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5)
beta=0.1; t0=time.time()
for ep in range(3):
    order=list(range(len(data))); random.shuffle(order); tot=0.0; acc=0
    for j in order:
        pl,ci,ri=data[j]
        pc=logp(ci,pl,True); prj=logp(ri,pl,True)
        rc=logp(ci,pl,False); rr=logp(ri,pl,False)
        margin=beta*((pc-rc)-(prj-rr)); loss=-F.logsigmoid(margin)
        opt.zero_grad(); loss.backward(); opt.step()
        tot+=loss.item(); acc+=margin.item()>0
    print("epoch %d  loss %.4f  chosen>rejected %d/%d  %.0fs" % (ep+1,tot/len(data),acc,len(data),time.time()-t0), flush=True)
m.save_pretrained(D+r"\qwen_lora_dpo")
print("adapter saved", flush=True)
m.eval()
@torch.no_grad()
def gen(p):
    e=tok(p,return_tensors="pt").to("cuda")
    o=m.generate(**e,max_new_tokens=90,do_sample=True,temperature=0.9,top_k=80,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][e["input_ids"].shape[1]:],skip_special_tokens=True).replace("\n"," ")
for p in ["黛玉含泪道","一日，众好汉","那道人手持宝剑"]:
    print("\n["+p+"] "+gen(p), flush=True)
open(D+r"\day4_done.flag","w").write("ok")
