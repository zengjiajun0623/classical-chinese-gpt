# classical-chinese-gpt

A hands-on project to understand how large language models are built, by training a tiny one from scratch on classical Chinese literature.

The training corpus is the simplified-Chinese text of the Four Great Classical Novels:

- 红楼梦 (Dream of the Red Chamber)
- 三国演义 (Romance of the Three Kingdoms)
- 水浒传 (Water Margin)
- 西游记 (Journey to the West)

All four are public-domain texts from Project Gutenberg. About 2.9 million characters in total, drawn from roughly 5,777 distinct characters.

## What is here

- `train_bigram.py` is a minimal baseline. It predicts the next character from only the current one. It trains in seconds and writes gibberish, but it shows the entire training loop (guess, measure the loss, adjust the numbers) with nothing hidden.
- `train_gpt.py` is a small character-level GPT: token and position embeddings, multi-head self-attention, and stacked transformer blocks. It runs the same training loop as the baseline, with attention added so the model can use context instead of a single character.

## Setup

Requires Python 3. The scripts use Apple Silicon's `mps` GPU backend; change the `device` line to `cpu` if you are not on an Apple Silicon Mac.

Install dependencies:

    python3 -m pip install torch
    brew install opencc

Download the four novels (traditional Chinese) from Project Gutenberg:

    curl -L https://www.gutenberg.org/cache/epub/24264/pg24264.txt -o hongloumeng.txt
    curl -L https://www.gutenberg.org/cache/epub/23950/pg23950.txt -o sanguo.txt
    curl -L https://www.gutenberg.org/cache/epub/23863/pg23863.txt -o shuihu.txt
    curl -L https://www.gutenberg.org/cache/epub/23962/pg23962.txt -o xiyouji.txt

Convert traditional to simplified Chinese:

    opencc -c t2s -i hongloumeng.txt -o hongloumeng_s.txt
    opencc -c t2s -i sanguo.txt -o sanguo_s.txt
    opencc -c t2s -i shuihu.txt -o shuihu_s.txt
    opencc -c t2s -i xiyouji.txt -o xiyouji_s.txt

## Run

    python3 train_bigram.py     # the baseline
    python3 train_gpt.py        # the transformer

Each script prints training and validation loss as it goes, with sample text so you can watch the writing improve.

## The dials

The hyperparameters at the top of `train_gpt.py` (context window, embedding size, number of attention heads, number of layers, learning rate, iterations) are the knobs that set how large and capable the model is. They are the same knobs used to train frontier models, set to laptop scale. Turn them up for better samples at the cost of longer training.

## Notes

The raw and converted text files are not committed. Run the setup steps above to reproduce them locally.
