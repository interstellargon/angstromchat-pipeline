"""
A function that help with evaluating a base model.
"""
import torch

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):