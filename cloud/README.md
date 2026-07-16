# cloud/ — GPT-2-scale training on a rented GPU

The scripts at the repo root (`train_bigram.py`, `train_gpt.py`, …) are the laptop-scale toys. This folder is the real thing: a ~124M-parameter GPT-2-scale model trained from scratch on Chinese Wikipedia on a rented RunPod RTX 4090, then fine-tuned toward classical Chinese. It needs a CUDA machine with `torch`, `datasets`, and `tokenizers`.

## Pipeline (run in order)

1. `dl_wiki.py` — stream Chinese Wikipedia (`wikimedia/wikipedia` 20231101.zh) into `zhwiki.txt` (~1.05B characters).
2. `tok_train.py` — train a 32k byte-level BPE tokenizer on a sample (kept small so it doesn't OOM). Output in `tok/` (committed here for reproducibility).
3. `tokenize_corpus.py` — tokenize the full corpus into `data.bin` (~656M tokens, uint16).
4. `train_gpt2.py` — pretrain: 12 layers, 768-dim, 12 heads (~124M params), block 512, bf16 + flash-attention (SDPA), cosine LR with warmup, gradient accumulation, checkpoints every 1000 steps → `gpt2_zh.pt`. About 2.2B tokens (~3.4 epochs), ~5–6h on one 4090.
5. `tokenize_classical.py` — tokenize the classical novels (the Gutenberg corpus from the repo root) → `classical.bin`.
6. `finetune.py` — fine-tune the base on a **70/30 classical/wiki blend** (style adaptation while avoiding catastrophic forgetting) → `gpt2_zh_classical.pt`.
7. `finetune_chain.sh` / `prep.sh` — orchestrators that wait for the prior stage's done-marker, then launch the next, so the whole run proceeds unattended.

## Scaling further

- `bench.py` — measures 4090 throughput at several model sizes, used to size a run to a budget via Chinchilla's ~20-tokens-per-parameter rule.
- `stage_fineweb.py` — streams FineWeb-2 (Mandarin, `cmn_Hani`) and tokenizes ~6B tokens of web text, the extra data needed to train a larger (~350M) model well.

## What's committed vs not

The `tok/` tokenizer (32k byte-level BPE) is committed. Data (`*.bin`, `zhwiki.txt`) and model checkpoints (`*.pt`) are **not**, they're large and fully regenerable from the steps above. Note the corpus is bilingual (Wikipedia mixes simplified and traditional), so the model reads/writes both; use an OpenCC pass at generation to force one script if desired.
