# Quick probe: ask an adapter the round-3 TRAINING questions verbatim.
# Distinguishes "memorized but can't generalize" from "didn't even store it".
#   py -3.10 cgpt\ttt_probe_3080.py --adapter qwen_ttt3b
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
QS = [
    "List the four CROPS properties named in the EF Mandate.",
    "What is the walkaway test in the EF Mandate?",
    "For what time horizon is the EF Mandate written?",
    "What is the Only-EF Rule?",
    "What is Subtraction as Success?",
]

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
m = PeftModel.from_pretrained(m, args.adapter)
m.eval()
for q in QS:
    t = tok.apply_chat_template([{"role": "user", "content": q}],
                                tokenize=False, add_generation_prompt=True)
    ids = tok(t, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=90, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    print("Q:", q, flush=True)
    print("A:", tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)[:250], flush=True)
    print(flush=True)
