#!/usr/bin/env python3
"""Train a byte-level BPE on WikiText-103 and save token-id .npy (datasets + tokenizers, no torch).

Gives an adequately-sized subword corpus (~15M tokens) so the combo can't trivially overfit, to
test whether the TinyShakespeare subword regression was just low-data overfitting.
"""
import os
import json
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
VOCAB = 8192
TRAIN_CHARS, VAL_CHARS = 60_000_000, 1_200_000

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

def take_chars(split, n):
    parts, total = [], 0
    for row in ds[split]:
        t = row["text"]
        parts.append(t)
        total += len(t) + 1
        if total >= n:
            break
    return ("\n".join(parts))[:n]

train_text = take_chars("train", TRAIN_CHARS)
val_text = take_chars("validation", VAL_CHARS)

tok = Tokenizer(BPE(unk_token="[UNK]"))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
trainer = BpeTrainer(vocab_size=VOCAB, special_tokens=["[UNK]"])
tok.train_from_iterator([train_text], trainer=trainer)

tr = np.array(tok.encode(train_text).ids, dtype=np.int64)
va = np.array(tok.encode(val_text).ids, dtype=np.int64)
np.save(os.path.join(D, "wt103_bpe_train.npy"), tr)
np.save(os.path.join(D, "wt103_bpe_val.npy"), va)
with open(os.path.join(D, "wt103_bpe_meta.json"), "w") as f:
    json.dump({"vocab": tok.get_vocab_size()}, f)
print(f"saved WT103 BPE | vocab={tok.get_vocab_size()} train_tokens={tr.size:,} val_tokens={va.size:,}")
