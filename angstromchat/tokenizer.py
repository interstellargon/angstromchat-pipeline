"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Andrej Karpathy's RustBPE Tokenizer for training and tiktoken for inference
"""

import rustbpe
import tiktoken
from functools import lru_cache
import os
import pickle
import torch

from angstromchat.common import get_base_dir


SPECIAL_TOKENS = [
    # every document begins with the BOS token
    "<|bos|>",
    # below tokens are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# this split pattern mainly deviates from GPT-4
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


# Tokenizer based on Andrej Karpathy's rustbpe and tiktoken combination
class RustBPETokenizer:
    """ Light wrapper around tiktoken (for efficient inference) but train with andrej karpathy's rustbpe"""
    
    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special > 256, f"vocab_size_no_special must be at least 256, but got {vocab_size_no_special}"

        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)

        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None and append is not None:
                ids = [prepend_id, *ids, append_id]
            elif prepend is not None:
                ids = [prepend_id, *ids]
            elif append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None and append is not None:
                ids = [[prepend_id, *ids_row, append_id] for ids_row in ids]
            elif prepend is not None:
                ids = [[prepend_id, *ids_row] for ids_row in ids]
            elif append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids        

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

def get_tokenizer():
    from angstromchat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}."
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes







        
