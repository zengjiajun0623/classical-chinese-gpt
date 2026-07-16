# Test-time training, train side. LoRA gradient steps on the raw document
# at "inference time" - no QA pairs, just next-token prediction on the doc,
# exactly like pretraining but on 60KB and for a few seconds.
#
#   py -3.10 cgpt\ttt_train_3080.py --steps 100 --out cgpt\qwen_ttt_100
#
# Also reports perplexity on a held-out slice of classical.txt before and
# after, so we can see catastrophic forgetting directly.
import argparse, os, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HERE = os.path.dirname(__file__)
DOC = os.path.join(HERE, "ef_mandate.txt")
HELDOUT = os.path.join(HERE, "classical.txt")  # forgetting probe, first 20k chars
CTX = 1024

def perplexity(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids[:, :4096].to("cuda")
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    return torch.exp(loss).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    random.seed(42)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")

    held = open(HELDOUT, encoding="utf-8").read()[:20000] if os.path.exists(HELDOUT) else None
    if held:
        print(f"held-out ppl BEFORE: {perplexity(model, tok, held):.2f}")

    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    doc_ids = tok(open(DOC, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
    print(f"document: {len(doc_ids)} tokens")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for step in range(args.steps):
        i = random.randint(0, max(0, len(doc_ids) - CTX - 1))
        chunk = doc_ids[i:i + CTX].unsqueeze(0).to("cuda")
        loss = model(chunk, labels=chunk).loss
        loss.backward()
        opt.step(); opt.zero_grad()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    model.eval()
    if held:
        print(f"held-out ppl AFTER:  {perplexity(model, tok, held):.2f}")
    model.save_pretrained(args.out)
    print("adapter saved to", args.out)

if __name__ == "__main__":
    main()
