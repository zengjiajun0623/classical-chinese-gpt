# classical-chinese-gpt

A hands-on project to understand how large language models are built, by training tiny ones from scratch on classical Chinese literature. It walks the full pretraining pipeline on a laptop: tokenize, train, watch it learn, then mix sources and steer the output style with control tokens.

## The corpus

Simplified-Chinese text of nine public-domain works from Project Gutenberg, spanning three registers:

- Ming-Qing vernacular novels: 红楼梦, 三国演义, 水浒传, 西游记, 儒林外史, 封神演义
- Classical 文言: 聊斋志异
- Modern vernacular: 鲁迅 (呐喊, 彷徨)

About 4.4 million characters drawn from roughly 6,700 distinct characters. Character-level tokenization (no sub-word merges needed for Chinese).

## The scripts

- `train_bigram.py` is a minimal baseline that predicts the next character from only the current one. It shows the full training loop (guess, measure loss, adjust) with nothing hidden.
- `train_gpt.py` is a small character-level GPT (multi-head self-attention, stacked transformer blocks) trained on the four core novels.
- `build_corpus2.py` builds the training corpus from all nine sources, tagging every short passage with its source so the model can learn to condition on it. It writes `train.txt` and `val.txt`, splitting each source 90/10 so validation represents every register.
- `train_tags.py` trains the GPT on the tagged corpus, with a fused-attention, GPU-resident data path for speed. Saves `model_tags.pt`.
- `generate.py` loads the trained model and "summons" a style by seeding with a source tag, for example `=== 鲁迅 ===`. It uses temperature and tag-banning to keep the chosen style locked.

## Setup

Requires Python 3 and PyTorch. The scripts use Apple Silicon's `mps` GPU backend; change the `device` line to `cpu` otherwise.

    python3 -m pip install torch
    brew install opencc

Download the nine texts (traditional Chinese) and convert them to simplified:

    curl -L https://www.gutenberg.org/cache/epub/24264/pg24264.txt -o hongloumeng.txt
    curl -L https://www.gutenberg.org/cache/epub/23950/pg23950.txt -o sanguo.txt
    curl -L https://www.gutenberg.org/cache/epub/23863/pg23863.txt -o shuihu.txt
    curl -L https://www.gutenberg.org/cache/epub/23962/pg23962.txt -o xiyouji.txt
    curl -L https://www.gutenberg.org/cache/epub/24032/pg24032.txt -o rulin.txt
    curl -L https://www.gutenberg.org/cache/epub/23910/pg23910.txt -o fengshen.txt
    curl -L https://www.gutenberg.org/cache/epub/51828/pg51828.txt -o liaozhai.txt
    curl -L https://www.gutenberg.org/cache/epub/27166/pg27166.txt -o nahan.txt
    curl -L https://www.gutenberg.org/cache/epub/24042/pg24042.txt -o panghuang.txt

    for f in hongloumeng sanguo shuihu xiyouji rulin fengshen liaozhai nahan panghuang; do
        opencc -c t2s -i $f.txt -o ${f}_s.txt
    done

## Run

    python3 train_gpt.py       # the single-corpus transformer
    python3 build_corpus2.py   # build the tagged multi-source corpus
    python3 train_tags.py      # train with source conditioning
    python3 generate.py        # summon a style by its tag

## Notes

Raw and converted text files, and trained model checkpoints, are not committed. Reproduce them with the steps above.
