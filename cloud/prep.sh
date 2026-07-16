#!/bin/bash
cd /workspace/build
exec > prep.log 2>&1
echo "[prep] waiting for wiki download to finish..."
for i in $(seq 1 240); do grep -q "DONE:" dl.log 2>/dev/null && break; sleep 15; done
grep -q "DONE:" dl.log || { echo "[prep] DOWNLOAD TIMEOUT/FAILED"; tail -5 dl.log; exit 1; }
echo "[prep] download ok: $(tail -1 dl.log)"; ls -lh zhwiki.txt
echo "[prep] training tokenizer..."
python3 -u tok_train.py || { echo "[prep] TOK FAILED"; exit 2; }
echo "[prep] tokenizing corpus..."
python3 -u tokenize_corpus.py || { echo "[prep] TOKENIZE FAILED"; exit 3; }
echo "[prep] COMPLETE"; ls -lh data.bin; touch PIPELINE_DONE
