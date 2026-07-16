# chinese-gpt-from-scratch

Building a Chinese language model from scratch, by hand, to genuinely understand how these things work. It began as a laptop experiment and grew into a real GPT-2-scale model trained on China's classical literature and Wikipedia. Every stage is in this repo: from a 4-million-parameter toy that writes gibberish, to a 124-million-parameter model that writes coherent Chinese.

This is a learning project, not a frontier model. The goal was never to compete with the big models. It was to build the entire pipeline myself, tokenizer, pretraining, fine-tuning, and understand every piece along the way. If you are learning the same thing, everything you need to reproduce it is here.

## The journey

### Stage 1: the laptop toy (repo root)

A character-level GPT trained on a Mac on the Four Great Classical Novels and more. It picks up the shape of Chinese (dialogue, character names, sentence rhythm), but at 4M parameters it is far too small to mean anything. It is perfect for *feeling* how tokenization, attention, and the training loop actually work.

- `train_bigram.py`: the simplest possible baseline, predict the next character from the current one.
- `train_gpt.py`: a real transformer (multi-head self-attention, stacked blocks) on the four novels.
- `build_corpus2.py`, `train_tags.py`, `generate.py`: a nine-source corpus with per-author style tags, so you can summon a specific author's voice at generation time.
- `build_sft.py`, `fine_tune.py`: supervised fine-tuning into a 问/答 chat format, the base-model-to-assistant leap in miniature.

### Stage 2: the real model ([`cloud/`](cloud/))

A ~124M-parameter GPT-2-scale model, trained from scratch on ~656M tokens of Chinese Wikipedia on a rented RTX 4090, then fine-tuned toward classical Chinese. This one writes genuinely coherent, grammatical Chinese. The full recipe is in [`cloud/README.md`](cloud/README.md).

### Stage 3: alignment ([`rlhf/`](rlhf/))

Teaching the model *taste*, using DPO (the run-it-on-one-GPU form of RLHF) on a local RTX 3080. The goal was to stop the model mixing characters across novels, and a human gave the preferences. It worked, then plateaued, then hit a wall when the judging was automated. The full, honest story (including a negative result worth more than a win) is in [`rlhf/README.md`](rlhf/README.md).

## The corpus

Two kinds of text, for two jobs:

- **Breadth (capability):** all of Chinese Wikipedia (~1 billion characters), so the model learns how language and the world generally work.
- **Style:** nine public-domain classics from Project Gutenberg, across three registers:
  - Ming-Qing vernacular novels: 红楼梦, 三国演义, 水浒传, 西游记, 儒林外史, 封神演义
  - Classical 文言: 聊斋志异
  - Modern vernacular: 鲁迅 (呐喊, 彷徨)

The toys normalize everything to simplified Chinese with OpenCC. The cloud model trains on raw (bilingual) Wikipedia and matches the prompt's script at generation with an OpenCC pass.

## Reproduce it

### The laptop toys

Requires Python 3 and PyTorch (Apple Silicon `mps`, or `cpu`).

```bash
python3 -m pip install torch
brew install opencc
```

Download the texts and convert them to simplified:

```bash
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
```

Then:

```bash
python3 train_gpt.py       # the char-level transformer on the four novels
python3 build_corpus2.py   # build the tagged multi-source corpus
python3 train_tags.py      # train with per-source style conditioning
python3 generate.py        # summon a style by its tag
```

### The real model

See [`cloud/`](cloud/). It needs a CUDA GPU and `torch`, `datasets`, `tokenizers`.

## Talking to it: one small experiment

Near the end, I asked the finished model, in Chinese, "who is Diaochan?" (a famous figure from Romance of the Three Kingdoms). It answered with fluent, grammatical, confident classical Chinese that contained no actual information about her. It did the same thing when asked any history question: it produced the perfect *shape* of an encyclopedia entry, filled with plausible-sounding nonsense.

That one failure holds three of the biggest lessons in the whole project:

- **It is a continuation engine, not an assistant.** A base model does not hear a question. It predicts what text usually follows that string. In a novel, a line of dialogue is followed by more dialogue, so that is what you get. Turning it into something that answers takes a further instruction-tuning step (question to answer format), which is the real gap between a base model and ChatGPT.
- **Format is not knowledge.** The model learned the form of an answer flawlessly and the facts not at all. Facts get into a model by being seen many times during pretraining, and a 124M model trained on a thin slice simply never stored them.
- **You cannot fine-tune facts in.** Fine-tuning on history text would only sharpen the confident tone while the knowledge stayed absent. That is training a more convincing hallucinator. Real knowledge needs a bigger model, more pretraining, or retrieval at question-time.

## What it taught me

Every hard part of training a language model shows up here at small scale: why tokenization is the hidden foundation, what overfitting really looks like, how a badly built validation split can fake a catastrophe, why data (not compute) is the real ceiling on model size, what fine-tuning can and cannot fix, why a shallow automatic reward gets hacked instead of satisfied, and why alignment polishes a base but cannot rescue one that was never capable. The models are tiny, but the lessons are exactly the ones the large labs run into.

## Notes

Raw text (`*.txt`), tokenized data (`*.bin`), and model checkpoints (`*.pt`) are not committed, they are large and fully regenerable from the steps above. The trained tokenizer is included.
