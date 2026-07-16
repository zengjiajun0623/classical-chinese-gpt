from datasets import load_dataset
print("loading Chinese Wikipedia (pulls ~1-2GB)...", flush=True)
ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train")
n = len(ds); print("articles:", n, flush=True)
out = "/workspace/build/zhwiki.txt"; chars = 0
with open(out, "w", encoding="utf-8") as f:
    for i, ex in enumerate(ds):
        t = ex["text"].strip()
        if t:
            f.write(t + "\n\n"); chars += len(t)
        if i % 100000 == 0:
            print(f"  {i}/{n} articles, {chars/1e6:.0f}M chars", flush=True)
print(f"DONE: {chars/1e6:.1f}M characters -> {out}", flush=True)
