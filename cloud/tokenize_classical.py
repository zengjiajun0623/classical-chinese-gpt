import numpy as np, glob
from tokenizers import ByteLevelBPETokenizer
tok=ByteLevelBPETokenizer("/workspace/build/tok/zhbpe-vocab.json","/workspace/build/tok/zhbpe-merges.txt")
eot=tok.token_to_id("<|endoftext|>")
def strip_g(t):
    a=t.find("*** START OF")
    if a!=-1: t=t[t.find("\n",a)+1:]
    b=t.find("*** END OF")
    if b!=-1: t=t[:b]
    return t.strip()
files=sorted(glob.glob("/workspace/build/classical/*_s.txt"))
print("files:",[f.split("/")[-1] for f in files],flush=True)
total=0
with open("/workspace/build/classical.bin","wb") as out:
    for f in files:
        buf=[]
        for line in strip_g(open(f,encoding="utf-8").read()).split("\n"):
            line=line.strip()
            if line: buf.append(line)
            if len(buf)>=2000:
                ch=[]
                for e in tok.encode_batch(buf): ch+=e.ids; ch.append(eot)
                np.array(ch,dtype=np.uint16).tofile(out); total+=len(ch); buf=[]
        if buf:
            ch=[]
            for e in tok.encode_batch(buf): ch+=e.ids; ch.append(eot)
            np.array(ch,dtype=np.uint16).tofile(out); total+=len(ch)
print("DONE classical tokens:",total,flush=True)
