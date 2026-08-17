"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on non-Hopper GPUs (including Blackwell), CPU.

Usage (drop-in replacement for FA3):
    from angstromchat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""

import torch
import os
from kernels import get_kernel

# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs

def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU). """
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper only. 
        if major != 9:
            return None        
        os.environ["HF_HUB_DISABLE_PROGRESS_BAR"] = "1"
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None

_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None