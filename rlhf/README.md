# Stage 3: alignment (RLHF, then RLAIF)

This is where the model stops just *writing* classical Chinese and starts writing the *kind* someone actually prefers. It runs on a local RTX 3080 and uses DPO (Direct Preference Optimization), the simpler cousin of RLHF that skips the separate reward model and trains straight on preference pairs.

The steering goal we picked was **"one novel's world per scene"**: if a passage opens in Dream of the Red Chamber, it should not wander into Water Margin characters. The base model mixes worlds constantly, because it was trained on all nine novels blended together with no signal to stay in one.

## The loop

1. **Generate** two (or more) continuations of the same prompt (`make_pairs_*.py`).
2. **Judge** which one is better. A human does this for real taste; a rule can do it at scale.
3. **Train** with DPO so the model raises the probability of the chosen text and lowers the rejected (`dpo*.py`, `rlaif_train_3080.py`).
4. **Re-generate** and measure whether the preference generalized to prompts it never saw judged.

## What actually happened (the honest version)

**RLHF, human-judged.** A first run on 6 hand-judged pairs *memorized*: the loss collapsed to zero in two epochs and it failed on held-out prompts. Scaling to 46 judgments with a gentler schedule and a reference-model leash *generalized*: brand-new prompts (李逵, 宝钗, 八戒, 周瑜) stayed in their own world. More judgments taught a principle instead of memorizing answers.

**The plateau.** The gains concentrated on prompts that carry a clear signal (a name that belongs to one novel). Genuinely ambiguous openers (道人, 真人, 老僧, which exist in three novels at once) stayed a coin-flip. That is not a training failure. It is the ceiling of the problem: there is no correct world for "a Taoist priest appeared," so no amount of feedback resolves it.

**RLAIF, machine-judged.** To scale past the human bottleneck, `rlaif_gen_*.py` encodes the "one world" rule as a character-to-novel lexicon and judges pairs automatically (ties, where both continuations stay in one world, are dropped as no-signal). This scaled to 119 preferences with no human in the loop. And then the honest result: on a 24-prompt held-out measurement, the RLAIF model scored **0.979 distinct-novels-per-passage against the base model's 0.979.** No improvement.

Why? The reward was a shallow proxy. "Minimize name-collisions" is not the same as "write good classical prose," so the model satisfied the letter of the rule on the training pairs without getting better on held-out text. That gap is **reward hacking**, the central failure mode of RLHF at every scale. A human catches things a lexicon cannot (a female character cannot have a beard); automate the judge and you automate its blind spots too.

## The takeaways

- DPO is RLHF you can run on one consumer GPU.
- Preference tuning teaches *behavior and taste*, not *knowledge*. It polishes a base; it cannot build one.
- More data helps only if the reward is faithful. A crude automatic reward scales the crudeness.
- Measure on held-out prompts, not the training set, and not on a single cherry-picked sample. The rigorous number will sometimes overturn the nice-looking anecdote.

## Files

| File | Job |
|------|-----|
| `make_pairs_3080.py` / `make_pairs2_3080.py` / `make_pairs3_3080.py` | generate A/B continuation pairs for human judging |
| `dpo_3080.py` / `dpo2_3080.py` / `dpo3_3080.py` | DPO training on human-judged pairs (6, then 26, then 46) |
| `rlaif_gen_3080.py` / `rlaif_gen2_3080.py` | generate + auto-judge pairs with the lexicon rule |
| `rlaif_train_3080.py` | DPO on the auto-labeled set plus the 6 real human picks |
| `gen_3080.py` | sample from any checkpoint with a repetition penalty |

All scripts load `gpt2_zh_classical.pt` (the Stage 2 fine-tuned model) and expect the tokenizer in `tok/`. They were run on Windows with `py -3.10`.

## Bonus: turning it into a question-answering assistant

A base model does not answer questions, it continues text. Ask it "谁是貂蝉？" and it just writes more novel. Two steps fix that.

**Respond (`sft_instruct_3080.py`).** Instruction-tuning: fine-tune on `问：… 答：…` pairs, computing loss only on the answer and ending each with the stop token. After this the model recognizes a question, answers it, and halts. But it now answers *everything* by copying the nearest memorized answer, so out-of-set questions hallucinate (ask about a character it was not taught and it grabs someone else's description).

**Be correct (`rag_sft_3080.py`, `rag_ask_3080.py`).** Retrieval-augmented generation. Keep a knowledge base of fact-cards (`knowledge.json`), retrieve the card for the entity in the question, and put it in front of the prompt: `参考：{card}\n问：{q}\n答：`. The model is fine-tuned to answer *from the card*. Ten characters were held entirely out of the answer-training and kept only as cards; the model answers them correctly anyway, by reading the retrieved card, which proves retrieval (not memory) is doing the work. When retrieval finds nothing, it was taught to say "我不知道" instead of bluffing.

The point: knowledge lives in the editable knowledge base, not the weights. To teach it a new fact, add a card. No retraining. That is how real assistants stay current, and it is the only way a 124M model can be reliably correct, because it is far too small to store facts in its parameters.
