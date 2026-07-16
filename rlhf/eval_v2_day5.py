"""Day 5: eval v2. Strict golds, invented-name honesty, sampled style.
Runs the Qwen ladder: base / +LoRA / +DPO / +RLAIF."""
import sys, time, torch
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
D=r"C:\Users\Jiajun\cgpt"; MID="Qwen/Qwen2.5-1.5B-Instruct"
tok=AutoTokenizer.from_pretrained(MID)
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

def run(tag,adapter):
    base=AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda")
    m=PeftModel.from_pretrained(base,D+"\\"+adapter) if adapter else base
    m.eval()
    @torch.no_grad()
    def chat(q,n=60):
        msgs=[{"role":"system","content":"你是简洁的中文助手。不知道就说“我不知道”。"},{"role":"user","content":q}]
        t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        e=tok(t,return_tensors="pt").to("cuda")
        o=m.generate(**e,max_new_tokens=n,do_sample=False,pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:],skip_special_tokens=True)
    @torch.no_grad()
    def cont(p):
        e=tok(p,return_tensors="pt").to("cuda")
        o=m.generate(**e,max_new_tokens=80,do_sample=True,temperature=0.95,top_k=100,pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][e["input_ids"].shape[1]:],skip_special_tokens=True)
    f=sum(any(g in chat(q) for g in gs) for q,gs in FACTS)
    h=sum(any(w in chat("谁是%s？"%e) for w in ["不知道","不确定","没有","未知","无法"]) for e in FAKE)
    s=0
    for p,w in STYLE:
        for _ in range(2):
            ws=worlds(cont(p))
            ok=(len(ws)<=1) if w=="无" else (ws<= {w})
            s+=ok
    print("%-12s facts %2d/20  honesty %2d/10  style %2d/16" % (tag,f,h,s), flush=True)
    del m,base; torch.cuda.empty_cache()

for tag,ad in [("qwen-base",None),("+LoRA","qwen_lora_classical"),("+DPO","qwen_lora_dpo"),("+RLAIF","qwen_lora_rlaif")]:
    run(tag,ad)
open(D+r"\day5_done.flag","w").write("ok")
