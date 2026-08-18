"""
GPT model 
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from dataclasses import dataclass
import torch
import torch.nn as nn

from angstromchat.common import print0


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6             # number of query heads
    n_kv_head: int = 6          # number of key/value heads (GQA)
    n_embd: int = 768           
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)    

class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        this __init__ function runs in meta device context.
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config

        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)

        # Tensor Cores achieve maximum efficiency when matrix dimensions are multiples of 8, 16, or 64.
        # Artificially expand the vocab_size to the nearest multiple of `pad_vocab_size_to` (e.g., 64).
        # The added 'dummy tokens' are safely cropped out in the forward() pass to keep outputs unchanged.
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab size from {config.vocab_size} to {padded_vocab_size} for hardware acceleration.")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

        # Per-layer learnable scalars (Inspired by modded-nanogpt)
        # 1. resid_lambdas: Controls the flow of the main residual stream.
        #    - Acts as a learnable "valve" to stabilize deep networks by scaling down 
        #      accumulated variance across layers. 
        #    - Initialized to 1.0 (neutral, 100% flow) and trained with a significantly 
        #      smaller learning rate to prevent abrupt destabilization of the network.        
        # 2. x0_lambdas: Continuously re-injects the original token embedding (x0) into every layer.
        #    - Prevents "representation collapse" (forgetting the original token meaning) 
        #      in deep layers and creates a direct gradient highway back to the input, 
        #      significantly accelerating training.
        #    - Initialized to 0.1 (blending 10% of x0) and trained more aggressively.
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()

        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})

        # To support meta device initialization, init the rotary embeddings here, but it's just "fake" meta tensors only.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        # Precision Note: "Compute in FP32, Store in BF16"
        # Why not use bfloat16 from the start? bfloat16 has a severely truncated mantissa 
        # (only 7 bits of precision). If we perform sensitive math operations—like division, 
        # large exponentiation (base ** x), and trigonometry (cos/sin)—in bfloat16, 
        # the microscopic rounding errors will exponentially snowball, completely 
        # corrupting the angular frequencies into garbage values.
        # Therefore, we safely compute all complex math in float32 to ensure numerical 
        # stability, and downcast to bfloat16 ONLY at the very end to save memory/bandwidth 
        # during the actual forward pass.
        cos, sin = cos.bfloat16, sin.bfloat16()
        # add batch and head dims for later broadcasting
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]     

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0)
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes
        

        

    
