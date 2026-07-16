# Round 2 of the TTT experiment, step 1: self-augmentation.
# The model reads the document chunk by chunk and generates its own study
# materials - QA pairs and paraphrases. No external teacher: same model,
# same document, just a smarter data format than raw next-token.
#
#   py -3.10 cgpt\ttt_augment_3080.py --out ttt_synth.jsonl
import argparse, json, os, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DOC = os.path.join(os.path.dirname(__file__), "ef_mandate.txt")
CHUNK_CHARS = 1500

QA_PROMPT = """Here is an excerpt from a document called "The Ethereum Foundation Mandate":

{chunk}

Write 5 question-answer pairs that test the specific facts in this excerpt.
Phrase each question the way a reader would ask about "the EF Mandate"
(standalone, no "this excerpt"). Keep answers short and factual.
Format exactly:
Q: ...
A: ...
"""

PARA_PROMPT = """Rewrite the following excerpt from "The Ethereum Foundation Mandate" in your own words. Keep every fact and every named term, change the phrasing:

{chunk}"""

def chunks(text):
    paras, buf, out = text.split("\n\n"), "", []
    for p in paras:
        if len(buf) + len(p) > CHUNK_CHARS and buf:
            out.append(buf.strip()); buf = ""
        buf += p + "\n\n"
    if buf.strip():
        out.append(buf.strip())
    return out

def gen(tok, model, prompt, temp=0.7):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=600, do_sample=True,
                             temperature=temp, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

def parse_qa(text):
    pairs = re.findall(r"Q:\s*(.+?)\s*\nA:\s*(.+?)(?=\nQ:|\Z)", text, re.S)
    return [(q.strip(), a.strip()) for q, a in pairs if len(q) > 10 and len(a) > 3]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ttt_synth.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="sdpa")
    model.eval()

    doc_chunks = chunks(open(DOC, encoding="utf-8").read())
    print(f"{len(doc_chunks)} chunks", flush=True)

    n_qa = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, ch in enumerate(doc_chunks):
            qa_text = gen(tok, model, QA_PROMPT.format(chunk=ch))
            pairs = parse_qa(qa_text)
            for q, a in pairs:
                f.write(json.dumps({"type": "qa", "q": q, "a": a}, ensure_ascii=False) + "\n")
            para = gen(tok, model, PARA_PROMPT.format(chunk=ch))
            f.write(json.dumps({"type": "text", "text": para}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"type": "text", "text": ch}, ensure_ascii=False) + "\n")
            n_qa += len(pairs)
            print(f"chunk {i+1}/{len(doc_chunks)}: {len(pairs)} QA pairs", flush=True)
    print(f"done: {n_qa} QA pairs + {2*len(doc_chunks)} text samples -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
