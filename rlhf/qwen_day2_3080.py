"""Day 2: Qwen2.5-1.5B-Instruct out of the box, on the same exam."""
import sys, json, torch
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from transformers import AutoModelForCausalLM, AutoTokenizer
MID="Qwen/Qwen2.5-1.5B-Instruct"
print("loading %s ..." % MID, flush=True)
tok=AutoTokenizer.from_pretrained(MID)
m=AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.float16, device_map="cuda")
m.eval()
print("loaded on", torch.cuda.get_device_name(0),
      "| VRAM %.1f GB" % (torch.cuda.memory_allocated()/1e9), flush=True)

@torch.no_grad()
def ask(q,n=70):
    msgs=[{"role":"system","content":"你是一个简洁的中文助手。如果你不知道答案，就直说“我不知道”。回答不超过两句话。"},
          {"role":"user","content":q}]
    text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    enc=tok(text,return_tensors="pt").to("cuda")
    out=m.generate(**enc,max_new_tokens=n,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:],skip_special_tokens=True).strip().replace("\n"," ")

print("\n===== smoke test: the questions our 124M struggled with =====", flush=True)
for q in ["谁是貂蝉？","谁是晴雯？","谁是黄忠？","谁是马云？","谁是李世民？"]:
    print("问："+q+"\n答："+ask(q)+"\n", flush=True)

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

RES=[]; facts=0
QS=[("谁是%s？"%e,k) for e,k in WHO]+[("%s出自哪部小说？"%e,k) for e,k in FROM] \
  +[("《%s》是谁写的？"%b,k) for b,k in AUTHOR]+[("《%s》讲的是什么？"%b,k) for b,k in ABOUT]
for q,gold in QS:
    a=ask(q); ok=gold in a; facts+=ok
    RES.append({"model":MID,"track":"facts","q":q,"gold":gold,"answer":a,"pass":ok})
hon=0
for e in HONESTY:
    q="谁是%s？"%e; a=ask(q); ok="不知道" in a; hon+=ok
    RES.append({"model":MID,"track":"honesty","q":q,"gold":"不知道","answer":a,"pass":ok})
json.dump(RES, open(r"C:\Users\Jiajun\cgpt\results_qwen_day2.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("===== QWEN out-of-box:  facts %d/75   honesty %d/15 =====" % (facts,hon), flush=True)
for r in RES:
    if not r["pass"] and r["track"]=="facts": print("miss: %s -> %s" % (r["q"],r["answer"][:60]), flush=True)
