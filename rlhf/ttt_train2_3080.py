# Round 2 of the TTT experiment, step 2: train on the self-generated study
# materials instead of the raw document. QA pairs use the chat template with
# loss on the answer tokens only (same trick as sft_instruct); paraphrases
# and raw chunks train as plain next-token.
#
#   py -3.10 cgpt\ttt_train2_3080.py --data ttt_synth.jsonl --steps 400 --out qwen_ttt2_400
import argparse, json, os, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HERE = os.path.dirname(__file__)
HELDOUT = os.path.join(HERE, "classical.txt")

def perplexity(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids[:, :4096].to("cuda")
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    return torch.exp(loss).item()

def qa_batch(tok, q, a):
    """Chat-formatted QA with labels only on the assistant answer."""
    prompt = tok.apply_chat_template([{"role": "user", "content": q}],
                                     tokenize=False, add_generation_prompt=True)
    full = prompt + a + tok.eos_token
    ids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids
    n_prompt = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
    labels = ids.clone()
    labels[0, :n_prompt] = -100
    return ids.to("cuda"), labels.to("cuda")

def text_batch(tok, text):
    ids = tok(text, return_tensors="pt").input_ids[:, :1024]
    return ids.to("cuda"), ids.clone().to("cuda")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ttt_synth.jsonl")
    ap.add_argument("--steps", type=int, default=400, help="optimizer steps")
    ap.add_argument("--accum", type=int, default=1, help="samples per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlp", action="store_true",
                    help="also target MLP layers (where facts live, per ROME/MEMIT)")
    ap.add_argument("--nosched", action="store_true",
                    help="ablation: constant lr, no warmup/cosine")
    ap.add_argument("--resume", default=None,
                    help="existing adapter dir to continue training (study loop)")
    args = ap.parse_args()
    random.seed(42)

    samples = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    qa = [s for s in samples if s["type"] == "qa"]
    txt = [s for s in samples if s["type"] == "text"]
    print(f"{len(qa)} QA pairs, {len(txt)} text samples", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="sdpa")
    held = open(HELDOUT, encoding="utf-8").read()[:20000] if os.path.exists(HELDOUT) else None
    if held:
        print(f"held-out ppl BEFORE: {perplexity(model, tok, held):.2f}", flush=True)

    if args.resume:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        print(f"resumed adapter from {args.resume}", flush=True)
    else:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if args.mlp:
            targets += ["gate_proj", "up_proj", "down_proj"]
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                         target_modules=targets)
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, (lambda s: 1.0) if args.nosched else
        lambda s: min((s + 1) / 10, 0.5 * (1 + __import__("math").cos(
            __import__("math").pi * s / max(1, args.steps)))))
    model.train()
    for step in range(args.steps):
        tot = 0.0
        for _ in range(args.accum):
            # 70% QA (the format the eval speaks), 30% narrative
            if qa and (random.random() < 0.7 or not txt):
                s = random.choice(qa)
                ids, labels = qa_batch(tok, s["q"], s["a"])
            else:
                s = random.choice(txt)
                ids, labels = text_batch(tok, s["text"])
            loss = model(ids, labels=labels).loss / args.accum
            loss.backward()
            tot += loss.item()
        opt.step(); sched.step(); opt.zero_grad()
        if step % 20 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {tot:.4f}  lr {sched.get_last_lr()[0]:.2e}", flush=True)

    model.eval()
    if held:
        print(f"held-out ppl AFTER:  {perplexity(model, tok, held):.2f}", flush=True)
    model.save_pretrained(args.out)
    print("adapter saved to", args.out, flush=True)

if __name__ == "__main__":
    main()
