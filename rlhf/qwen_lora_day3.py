"""Day 3: QLoRA-style fine-tune of Qwen2.5-1.5B on the classical corpus.
LoRA adapters (r=16) on a frozen fp16 base; ~300 steps; saves adapter + samples."""
import sys, time, random, torch
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
random.seed(0); D=r"C:\Users\Jiajun\cgpt"
MID="Qwen/Qwen2.5-1.5B-Instruct"
tok=AutoTokenizer.from_pretrained(MID)
m=AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda")
cfg=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"])
m=get_peft_model(m,cfg); m.print_trainable_parameters()

text=open(D+r"\classical.txt",encoding="utf-8").read()
print("corpus: %.1fM chars" % (len(text)/1e6), flush=True)
ids=tok(text[:6_000_000], return_tensors=None)["input_ids"]
print("tokens: %.1fM" % (len(ids)/1e6), flush=True)
CTX=512; STEPS=300; ACC=4
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
m.train(); t0=time.time()
for step in range(STEPS):
    opt.zero_grad()
    for _ in range(ACC):
        i=random.randrange(0,len(ids)-CTX-1)
        x=torch.tensor([ids[i:i+CTX]],device="cuda")
        loss=m(input_ids=x, labels=x).loss/ACC
        loss.backward()
    opt.step()
    if (step+1)%50==0:
        print("step %3d  loss %.3f  %.0fs" % (step+1, loss.item()*ACC, time.time()-t0), flush=True)
m.save_pretrained(D+r"\qwen_lora_classical")
print("adapter saved", flush=True)
m.eval()
@torch.no_grad()
def gen(p):
    enc=tok(p,return_tensors="pt").to("cuda")
    out=m.generate(**enc,max_new_tokens=100,do_sample=True,temperature=0.9,top_k=80,
                   pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:],skip_special_tokens=True).replace("\n"," ")
for p in ["话说贾宝玉","却说曹操引兵","那大圣按落云头","武松提了哨棒"]:
    print("\n["+p+"] "+gen(p), flush=True)
open(D+r"\day3_done.flag","w").write("ok")
