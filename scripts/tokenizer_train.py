"""
Train a tokenizer using Andrej Karpathy's nanochat BPE Tokenizer library.
In the style of GPT-4 tokenizer
"""

import argparse
import time
import os
import torch

from angstromchat.dataset import parquets_iter_batched
from angstromchat.tokenizer import RustBPETokenizer
from angstromchat.common import get_base_dir


parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum number of characters to train on')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum number of characters per document')
parser.add_argument('--vocab-size', type=int, default=32_768, help='Target vocabulary size')
args = parser.parse_args()
print(f"""Training tokenizer with parameters:
Max chars: {args.max_chars:,}
Doc cap: {args.doc_cap:,}
Vocab size: {args.vocab_size:,}
""")

# Text iterator
def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when exceeded args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(split='train'):
        for doc in batch:
            if len(doc) > args.doc_cap:
                doc = doc[:args.doc_cap]
            nchars += len(doc)
            yield doc
            if nchars > args.max_chars:
                return

text_iter = text_iterator()

# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Tokenizer training time: {train_time:.2f}s")

# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, 'tokenizer')
tokenizer.save(tokenizer_dir)

# Quick inline sanity check
test_text = """[Tokenizer Sanity Check: Level 99] 🚀
1. Multilingual: 안녕하세요, 세상! 🌍 Hello, world! こんにちは!
2. Spacing & Indentation:
\tif (True):
\t\tprint("Tabs\tand multiple     spaces check.")
3. Numbers & Math: e = 2.71828, 1,000,000 params, cost is $42.99!
4. Code & Symbols: [{`"key"`: ~@#$%^&*()_+=\\|/?<>,.}]
5. Edge cases & URLs: https://github.com/test-repo/tokenizer?ref=main&val=1
6. Contractions: I've, shouldn't've, y'all'd've.
7. Complex Unicode: 👨‍👩‍👧‍👦 (Family emoji), 한글자모 ㄲㄸㅃㅆㅉ
"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# Cache Token ID to Byte length mapping for efficient Bits Per Byte (BPB) calculation.
# BPB is a primary metric that provides a vocab-size-invariant loss for objective evaluation.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings(token_id)
    if token_str in special_set:
        token_bytes.append(0)
    else:
        id_bytes = len(token_str.encode("utf-8"))  # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
from angstromchat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args),
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    } 
])



