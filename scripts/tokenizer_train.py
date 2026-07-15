import argparse
import time
import os

from angstromchat.dataset import parquets_iter_batched
from angstromchat.tokenizer import RustBPETokenizer
from angstromchat.common import get_base_dir


"""
Train a tokenizer using Andrej Karpathy's nanochat BPE Tokenizer library.
In the style of GPT-4 tokenizer
"""

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




