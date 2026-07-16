import os
os.environ.setdefault("RAYON_NUM_THREADS","24")   # leave cores for training
import numpy as np
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
tok = ByteLevelBPETokenizer("/workspace/build/tok/zhbpe-vocab.json","/workspace/build/tok/zhbpe-merges.txt")
eot = tok.token_to_id("<|endoftext|>")
TARGET = 6_000_000_000
print("streaming FineWeb-2 cmn_Hani -> fineweb.bin (target %dB tokens)" % (TARGET//10**9), flush=True)
ds = load_dataset("HuggingFaceFW/fineweb-2", name="cmn_Hani", split="train", streaming=True)
total=0; buf=[]; next_log=100_000_000
def flush(buf,out):
    chunk=[]
    for e in tok.encode_batch(buf): chunk+=e.ids; chunk.append(eot)
    np.array(chunk,dtype=np.uint16).tofile(out); return len(chunk)
with open("/workspace/build/fineweb.bin","wb") as out:
    for ex in ds:
        t=ex.get("text","")
        if not t: continue
        buf.append(t)
        if len(buf)>=1000:
            total+=flush(buf,out); buf=[]
            if total>=next_log:
                print("%.2fB tokens staged" % (total/1e9), flush=True); next_log+=100_000_000
            if total>=TARGET: break
    if buf: total+=flush(buf,out)
print("DONE fineweb %d tokens (%.2fB)" % (total,total/1e9), flush=True)
