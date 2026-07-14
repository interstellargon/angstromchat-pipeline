import argparse
import time

from angstromchat.dataset import parquets_iter_batched

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
