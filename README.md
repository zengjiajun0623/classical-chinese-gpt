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

The same directory also has the **question-answering** step: instruction-tuning so the base model *responds* to questions instead of continuing text, then retrieval (RAG) so its answers are actually *correct*, then declining ("我不知道") instead of bluffing. Knowledge ends up in an editable knowledge base, not the weights, which is the only way a 124M model can be reliably correct. Details in [`rlhf/README.md`](rlhf/README.md).

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

## Week 2: the same stack on a real base (Qwen2.5-1.5B)

The from-scratch model proved the pipeline; week 2 reran it the way practitioners do, adapting a capable open base on one RTX 3080. Eval v2 (strict golds, invented-name honesty, sampled style):

| model | facts /20 | honesty /10 | style /16 |
|---|---|---|---|
| qwen-base | 15 | 0 | 13 |
| + classical LoRA (4 min, 0.28% params) | 17 | 0 | 13 |
| + DPO on 46 human judgments | 16 | 0 | 14 |
| + on-policy RLAIF | 14 | 0 | 14 |

What it taught: capability comes from the base and fine-tuning only steers it; LoRA added style with no forgetting; preference tuning improved style while taxing factual QA (the alignment tax, measured); every variant hallucinates about invented people (honesty 0/10), because helpfulness training runs deeper than any of our tuning; and on-policy preference pairs (9 of them) fixed a world-bleed that 119 stale off-policy pairs could not. Scripts in `rlhf/qwen_*.py`, eval in `rlhf/eval_v2_day5.py`.

### The full ladder (eval v2, all models)

| model | facts /20 | honesty /10 | style /16 |
|---|---|---|---|
| 124M SFT-chat | 20 * | 0 | - |
| 124M RAG (cards) | 15 | **10** | - |
| 124M classical / DPO-3 | - | - | 16 † |
| qwen-base | 15 | 0 | 13 |
| qwen +LoRA | 17 | 0 | 13 |
| qwen +DPO / +RLAIF | 16 / 14 | 0 | 14 |

\* Contaminated: all 20 questions were in its 52 training pairs, so this is recitation, not knowledge. † The style metric counts named-character collisions; the 124M writes vaguer prose with fewer names, so it fails less. Both are examples of the project's recurring lesson: every metric lies somewhere, and reading the raw rows is the only defense. The one clean 124M win is RAG honesty 10/10 (vs 0/10 for every Qwen variant): declining enforced by retrieval design beats declining hoped for from training.

## Week 3: test-time training, or teaching a model a document it has never seen

Andrej Karpathy's "preview of things to come" slide lists **test-time training?**, with the question mark in the original: models today are frozen at inference, and everything they learn in a conversation evaporates with the context window. Could a model instead *learn* a document, at inference time, into its weights? Earlier in this README I wrote "you cannot fine-tune facts in." Week 3 put that claim under a proper microscope. It survived three rounds and fell, with conditions, in the fourth.

**The setup.** The test document is the Ethereum Foundation's Mandate (published July 2026, after Qwen2.5's training cutoff, so provably unseen; ~12k tokens of text extracted from the PDF). An eval asks 14 keyword-scored factual questions about it, 3 honesty questions about invented terms that sound like they belong ("the Meadow Protocol"), and 3 general-knowledge questions as a forgetting probe. Every training arm also measures perplexity on held-out classical Chinese before and after, so forgetting shows up as one number. Closed-book baseline: **0/14**, which is the point, any gain is attributable to what happens at test time. Reading the whole document in context scores **12/14** and is the ceiling to beat.

| arm | facts /14 | honesty /3 | held-out ppl |
|---|---|---|---|
| closed book | 0 | 3 | 23.90 |
| in-context (whole doc in prompt) | **12** | 2 | unchanged |
| R1: LoRA on raw text, 400 steps | 1 | 1 | 33.92 (+42%) |
| R2: LoRA on self-generated QA cards | 0 | 0 | 24.85 |
| R3: LoRA on frontier-model QA cards | 0 to 1 | 0 | 24.50 |
| R4: all four fixes below | **12** | **3** | 26.50 (+11%) |

**Round 1, raw next-token training,** was the naive reading of the idea: run LoRA gradient steps on the document itself. Training loss fell to 0.5, the model could nearly recite the mandate, and it answered questions about it at 1/14 while paying 42% held-out perplexity. Memorized continuation is not queryable knowledge. It answered "what does CROPS stand for" with a confident, mandate-flavored, wrong expansion: it had learned the *style* and not the facts. Honesty collapsed too: knowing the document exists made it confabulate about invented terms instead of declining.

**Round 2, self-study,** had the model generate its own QA flashcards from each chunk, then train on those with loss on the answers only. Score: 0/14, *worse* than rote. Two failures compounded: a 1.5B model writes vague, excerpt-referential cards ("what does the excerpt state about..."), and QA-format training without real facts underneath just teaches confident short-answer *behavior*. It became a fluent fabricator.

**Round 3, hiring a tutor,** swapped in 88 precise QA cards written by a frontier model (Claude). Still ~0/14, but the failure got interesting. With clean optimization (batched gradients, cosine decay) the model answered its verbatim training questions perfectly ("List the four CROPS properties" gave the exact right list) yet failed trivial paraphrases of the same facts. Knowledge injected this way is **phrasing-locked**: a surface mapping from one question to one answer, not a fact reachable from any direction.

**Round 4 stacked four fixes, and together they worked:**

1. **Diversity per fact.** ~10 surface forms per atomic fact (question variants, declarative statements, prefixed contexts) instead of 1 or 2. This is the synthetic-continued-pretraining recipe: facts become robust only after appearing in many guises.
2. **Write to the right memory.** Earlier rounds' LoRA touched only attention projections. Interpretability work (ROME, MEMIT) locates factual associations in the MLP layers, so round 4 added gate/up/down projections to the LoRA targets (18.5M trainable params vs 4.4M).
3. **Teach the boundary, not just the contents.** 48 refusal cards about invented-but-plausible terms, answered "the Mandate doesn't mention that." Refusal then *generalized*: the eval's invented terms are different from the deck's, and the model declined them anyway.
4. **Optimizer hygiene.** Gradient accumulation of 8 with warmup and cosine decay. Round 3's single-sample steps left loss oscillating between 0.3 and 4.0; round 4 converged smoothly.

Result: **12/14 facts from bare weights, matching the in-context ceiling, with 3/3 honesty (the best of any arm, in-context included) and general knowledge intact,** at a cost of 11% held-out perplexity and the teacher tokens to write the deck. The phrasing generalization is genuine: the eval's wordings were deliberately kept out of the deck. Honest caveats: the same frontier model wrote both the study deck and the eval, so coverage of the test's content was total by design; and rounds 3 and 4 are distillation rather than pure self-contained TTT, which is also exactly what a production system would do (a big model preprocesses documents into study material for a small local one).

What week 3 taught, in one line each:

- Next-token exposure teaches recitation; retrieval needs the fact seen from many angles. "Format is not knowledge" cuts both ways.
- A small model cannot write its own study materials; card quality is a capability, not a formality.
- Naively injected facts are phrasing-locked; diversity is what unlocks them.
- Honesty is a boundary you must teach explicitly, or domain-training turns the model into a confident hallucinator about everything near the domain.
- Reading (in-context, RAG) is still the cheap, safe default. Weights-writing became competitive only with all four fixes, and it still taxes the base model. Karpathy's question mark is well earned.

Scripts: `rlhf/ttt_eval_3080.py` (three-arm eval; its `DocCache` prefills the document's KV cache once and reuses it per question, cutting the in-context eval from ~30 minutes to 2), `rlhf/ttt_train_3080.py` (round 1), `rlhf/ttt_augment_3080.py` (round 2 self-study), `rlhf/ttt_synth_fable.py` / `rlhf/ttt_synth_fable2.py` (tutor decks), `rlhf/ttt_train2_3080.py` (QA trainer, `--accum`, `--mlp`), `rlhf/ttt_probe_3080.py` (verbatim-vs-paraphrase probe). Result JSONs in `rlhf/ttt_results_*.json`. The document text is regenerated with `curl https://ethereum.foundation/ef-mandate.pdf | pdftotext`.

### The extension: ablation, a blind replication, and a self-study loop

The 12/14 result had two soft spots: I did not know which of the four fixes actually carried it, and the same model had written both the study deck and the exam. The same evening ran three follow-ups.

**Ablation (leave one fix out, retrain, re-score).**

| removed | facts /14 | honesty /3 | held-out ppl |
|---|---|---|---|
| nothing (full recipe) | 12 | 3 | +11% |
| phrasing diversity | 8 | 3 | +10% |
| refusal cards | **14** | **0** | +10% |
| MLP LoRA targets | **5** | 3 | +11% |
| optimizer hygiene | **14** | **3** | **+22%** |

The recipe is not four goods; it is two knowledge carriers and two dials. MLP placement is the backbone (attention-only LoRA loses 7 of 12 facts) and diversity is second (2 phrasings per fact loses 4). The refusal cards turn out to be a **knowledge-honesty dial**: remove them and the model scores a perfect 14/14 while confabulating about every invented term; keep them and the boundary occasionally swallows a real fact (that is exactly where round 4's two misses went, it refused the real Only-EF Rule). And optimizer hygiene is a **forgetting dial**, not a knowledge ingredient: the crude version also scores 14/14 with honesty intact, it just wrecks twice as much of the model's prior knowledge doing it.

**Blind replication.** Two fresh agents, neither allowed to see any of my artifacts, each read only the document: one wrote a 44-fact study deck (`rlhf/ttt_deck_indep.jsonl`), the other a 20-question exam (`rlhf/ttt_questions_indep.json`). On the blind exam: closed book 0/20, reading the document in context 19/20, my original round-4 adapter **8/20** (the coverage gap made visible: it knows what its deck taught and little else), and an adapter trained on the blind deck **13/20**. So under honest conditions the recipe delivers about two thirds of the reading ceiling, not parity; the 12/14-equals-ceiling result above was partly the experimenter tuning against his own test. This is the number I trust.

**Self-study loop (negative result).** The blind adapter's five misses were all facts present in its deck, so the obvious loop is automatic: practice-test the model on fresh phrasings of the deck's facts, find what did not stick, upweight exactly those cards (with replay of the rest), resume training gently, repeat. One iteration made things *worse*: practice 19/30 to 17/30, blind exam 13/20 to 11/20. Reinforcing weak memories destabilized strong ones. Consolidation without interference is an unsolved problem even at flashcard scale, and any real test-time-training system needs an answer to it.

**Where this leaves the question mark.** Per-document, writing into weights is dominated by reading on every axis: a document costs one cacheable prefill to read, versus a teacher's worth of tokens plus thousands of training passes to inject, for fewer correct answers, plus a forgetting tax that compounds across documents, plus a standing prompt-injection surface (anything that writes into weights is an attack channel; the honesty collapses in rounds 1-3 were tame previews). The versions that do make sense change the unit from a document to a growing corpus: the slow, heavily curated global flywheel that labs already run on deployment feedback, and the personal one, a local model absorbing months of one user's own notes and corrections on the user's own GPU, where diversity accrues naturally, there is no adversary, and the forgetting tax stays contained in a private adapter. That last one is a 3080-class workload, and this repo now contains the recipe for it.

One line for the whole week: we made writing-into-weights work well enough to measure exactly why reading wins.
