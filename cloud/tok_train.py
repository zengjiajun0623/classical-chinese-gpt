from tokenizers import ByteLevelBPETokenizer
import os
tok = ByteLevelBPETokenizer()
tok.train(files=["/workspace/build/zhwiki.txt"], vocab_size=32000, min_frequency=2,
          special_tokens=["<|endoftext|>"])
os.makedirs("/workspace/build/tok", exist_ok=True)
tok.save_model("/workspace/build/tok", "zhbpe")
print("DONE tokenizer vocab", tok.get_vocab_size(), flush=True)
