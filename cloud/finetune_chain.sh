#!/bin/bash
cd /workspace/build; exec > finetune.log 2>&1
echo "[ft] waiting for pretraining (TRAINING_DONE)..."
while [ ! -f TRAINING_DONE ]; do sleep 60; done
echo "[ft] pretraining done:"; ls -lh gpt2_zh.pt
echo "[ft] tokenizing classical corpus..."
python3 -u tokenize_classical.py || { echo "[ft] CLASSICAL TOKENIZE FAILED"; exit 2; }
echo "[ft] fine-tuning base on classical..."
python3 -u finetune.py || { echo "[ft] FINETUNE FAILED"; exit 3; }
echo "[ft] FINETUNE COMPLETE"; ls -lh gpt2_zh_classical.pt; touch FINETUNE_DONE
