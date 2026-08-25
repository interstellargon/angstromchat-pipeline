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

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding, Output Projection
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks
        n_embd = self.config.n_embd
        # Set bounds for Uniform initialization to avoid outliers.
        # Multiplying by sqrt(3) ensures the Uniform distribution U(-s, s) 
        # exactly matches the ideal standard deviation (1 / sqrt(n_embd)) of a Normal distribution.
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            # Zero-init prevents initial noise corruption and forces a 1-step gradient delay, stabilizing early training.
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like attn.c_v)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Initialize gate weights to 0 so the initial gate value becomes 2 * sigmoid(0) = 1.0.
        # This neutrally injects 100% of the Value Embedding at the start of training
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

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
        cos, sin = cos.bfloat16(), sin.bfloat16()
        # add batch and head dims for later broadcasting
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]   
        return cos, sin  

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

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        """
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total
        }

    def estimate_flops(self):
        """
        FLOPs Calculus for Transformer Language Model Training (Forward + Backward)
        ===========================================================================

        1. Summary Formula (Per Token)
        ------------------------------
        num_flops_per_token = 6 * N_matmul + sum_{l=1}^L (12 * h * q * effective_seq_len_l)

        Where:
            - N_matmul            : Total matmul weight parameters (excluding embedding layer)
            - L                   : Number of transformer layers (n_layer)
            - h                   : Number of attention heads (n_head)
            - q                   : Dimension per head (head_dim = d_model / n_head)
            - effective_seq_len_l : Effective sequence length at layer l (accounts for sliding windows)


        2. Detailed Breakdown & Derivation
        ----------------------------------
        (1) Weight Parameter FLOPs: 6 * N_matmul    
            Assuming all weight parameters perform dense matrix multiplications with input tensors.

            - Forward Pass (2 FLOPs / parameter):
            Computing matrix multiplication Y = X @ W requires 1 multiplication (*) and 1 addition (+)
            per inner-product step for each weight parameter.
            => Forward FLOPs = 2 * N_matmul

            - Backward Pass (4 FLOPs / parameter, 2x Forward):
            Requires computing two chain-rule gradients:
                1) dL/dX = dL/dY @ W^T    -> 2 * N_matmul FLOPs
                2) dL/dW = X^T @ dL/dY    -> 2 * N_matmul FLOPs
            => Backward FLOPs = 4 * N_matmul

            - Combined Total (Forward + Backward):
            2 * N_matmul (Forward) + 4 * N_matmul (Backward) = 6 * N_matmul FLOPs

            * Note on Embedding Layer (wte):
            Token embedding is an index-based memory lookup (Table Lookup), not a matrix 
            multiplication. Since it involves no floating-point arithmetic, it is excluded:
            N_matmul = N_total - N_embedding.


        (2) Attention Mechanism FLOPs: 12 * h * q * effective_seq_len  
            Direct matrix multiplications between dynamically generated activation tensors during 
            the self-attention process, independent of weight parameters

            - Q @ K^T  (Query-Key dot product)
            - P @ V    (Attention probability @ Value product, where P = Softmax(Q @ K^T / sqrt(q)))

            For a single layer across a sequence length T:
            - Q @ K^T Forward : (h, T, q) @ (h, q, T) -> (h, T, T) => 2 * h * q * T^2 FLOPs
            - P @ V   Forward : (h, T, T) @ (h, T, q) -> (h, T, q) => 2 * h * q * T^2 FLOPs
            - Total Forward   = 2 * h * q * T^2 + 2 * h * q * T^2 = 4 * h * q * T^2 FLOPs
            - Total Backward  = 2 * (Forward FLOPs) = 8 * h * q * T^2 FLOPs
            - Total (Fwd+Bwd) = 4 * h * q * T^2 + 8 * h * q * T^2 = 12 * h * q * T^2 FLOPs (per sequence)

            Per-Token FLOPs Conversion:
            Dividing total sequence FLOPs by T gives per-token FLOPs for 1 layer:
                Attn FLOPs / token = (12 * h * q * T^2) / T = 12 * h * q * T
            (Note: h * q = d_model, so this can also be expressed as 12 * d_model * T)

            * Sliding Window Attention Adaptation:
            Full sequence length T is capped by window size W_l at layer l:
            effective_seq_len_l = min(T, W_l).


        (3) Omitted Minor Operations
            Non-matmul operations (<1% of total training compute) are omitted for clarity:
            - Softmax (exp, sum, division)
            - Normalization layers (RMSNorm / LayerNorm)
            - Activation functions (GELU, SwiGLU, ReLU^2)
            - Element-wise operations (Bias addition, Residual connections)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel + 
                           self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        # Sum attention FLOPs per layer
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]     # (left, right) tuple, we only use left
            effective_seq = t if window < 0 else min(t, window)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token
        

        

        

    
