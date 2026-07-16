import numpy as np
from tokenizers import ByteLevelBPETokenizer
tok = ByteLevelBPETokenizer("/workspace/build/tok/zhbpe-vocab.json","/workspace/build/tok/zhbpe-merges.txt")
eot = tok.token_to_id("<|endoftext|>")
total = 0; buf = []
def flush(buf, out):
    chunk = []
    for e in tok.encode_batch(buf):
        chunk += e.ids; chunk.append(eot)
    np.array(chunk, dtype=np.uint16).tofile(out)
    return len(chunk)
with open("/workspace/build/data.bin","wb") as out:
    with open("/workspace/build/zhwiki.txt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            buf.append(s)
            if len(buf) >= 4000:
                total += flush(buf, out); buf = []
        if buf: total += flush(buf, out)
print("DONE tokenize %d tokens -> data.bin" % total, flush=True)
