"""
Pretrain base model
"""

import argparse
import torch
import torch.nn as nn
import wandb
from dataclasses import asdict
import json
import os
from contextlib import contextmanager
import math

from angstromchat.common import autodetect_device_type, print0, compute_init, get_peak_flops, DummyWandb, get_base_dir
from angstromchat.flash_attention import HAS_FA3
from angstromchat.tokenizer import get_tokenizer, get_token_bytes
from angstromchat.gpt import GPTConfig, GPT
from angstromchat.checkpoint_manager import load_checkpoint


# -----------------------------------------------------------------------------
# CLI arguments

parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu (empty = autodetect)")
# Compile
parser.add_argument("--no-compile", action="store_true", help="disable torch.compile to save VRAM")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise(faster, recommended) or rowwise(more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect_ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window attention pattern tiled across layers: L=Attention can view the entire sequence length, S=Attention is restricted to a sliding window of half the sequence length to save memory/computation")
# Training horizon (The program applies only one of these settings, prioritizing them in the order listed)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. if OOM on VRAM, reduce to 16,8,4,... .")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  

# -----------------------------------------------------------------------------
# Initialize hardware/DDP configuration, wandb logging, Flash Attention optimizations, and the tokenizer for model setup.

# hardware/DDP configuration
print0(f"\n\n---------------------------------- START ANGSTROMCHAT ----------------------------------")
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="angstromchat", name=args.run, config=user_config)

# Flash Attention status
if HAS_FA3:
    print0("✓ Using Flash Attention 3.")
else:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")

# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes()
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Init model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    base_dim = depth * args.aspect_ratio                                                    # base_dim = 16 * 64 = 1024
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim           # model_dim = 1024
    num_heads = model_dim // args.head_dim                                                  # num_heads = 1024 // 128 = 8
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern
    )

    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth)    # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device)           # 2) All tensors get storage on target device but with uninitialized garbage data
model.init_weights()                    # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data  # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA")
    else:
        from angstromchat.fp8 import Float8LinearConfig, convert_to_float8_training

        # Filter: only convert layers with dimensions divisible by 16 (FP8 hardware requirement)
        def fp8_module_filter(mod: nn.Module, fqn:str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8_layers = sum(1 for module in model.modules() if 'Float8' in type(module).__name__) 
        num_skipped = sum(1 for module in model.modules() if isinstance(module, nn.Linear)) - num_fp8_layers  
        print0(f"Successfully converted {num_fp8_layers} layers to FP8, skipped {num_skipped} layers (dimensions not divisible by 16).") 

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation."""

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield   # Don't exist FP8 modules -> nothing to do
        return

    # Swap Float8Linear -> nn.Linear (shares the same weight tensor, no copy)
    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype
        )
        linear.weight = fp8_module.weight   # share, no copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore all Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model (optional; disable with --no-compile to save VRAM)

orig_model = model  # original, uncompiled model, for saving raw model state_dict and for inference/evaluation
if not getattr(args, "no_compile", False):
    model = torch.compile(model, dynamic=False) # inputs' shape doesn't change across iterations so dynamic=False is suitable
else:
    print0("--no-compile: using eager mode (saves VRAM, slower training)")

# -----------------------------------------------------------------------------
# Scaling Laws Hyperparameter Transfer (muP-style)
# Calculates the optimal batch size, learning rate, and weight decay for the user-specified target model. 
# This is achieved by applying scaling laws(Power Lines, T_epoch) to extrapolate from the empirically tuned baseline hyperparameters of a small reference model (Depth 12).

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for name, count in param_counts.items():
    print0(f"{name:23s}: {count:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Determine the compute-optimal training horizon (total target tokens D) via scaling laws.
# Scaling laws assume a fixed target token-to-parameter ratio (D = ratio * N_scaling).
# To maintain linear scaling trends, N_scaling is strictly defined as core transformer weight matrices plus the lm_head.
def get_scaling_params(model):
    params_counts = model.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
# Optimal tokens for the model we are about to train
target_tokens = int(args.target_param_data_ratio * num_scaling_params)

# Reference model (Depth 12): Hyperparameters empirically tuned at this scale serve as the baseline extrapolated to higher depths (muP style).
d12_ref = build_model_meta(12)  # creates the model on meta device
D_REF = int(args.target_param_data_ratio * get_scaling_params(d12_ref))  # compute-optimal d12 training horizon in tokens 
B_REF = 2**19   # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Compute-Optimal Batch Size Scaling (Power Lines: B_opt ∝ D^0.383)
# Ref: https://arxiv.org/abs/2505.13738
# As the token horizon (D) expands, scaling batch size linearly causes FLOP waste via data redundancy, whereas a fixed batch size leads to gradient noise thrashing in low-loss regimes.
# The sub-linear exponent (0.383) defines the mathematical equilibrium for compute allocation:
# investing ~38% of the scaled budget into spatial noise suppression (larger batch) and ~62% into temporal loss landscape traversal (more update steps).
# The calculated optimal value is then clamped to the nearest power of 2 in logarithmic scale to guarantee memory alignment and maximize hardware utilization (MFU).
total_batch_size = args.total_batch_size    # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))  # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Calculate a learning rate scaling factor based on batch size (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF 
if batch_ratio != 1.0:
    # The core logic behind Square Root Scaling (η ∝ √(B/B_ref)):
    # 1. As batch size (B) increases, the variance (noise) of the mini-batch gradient decreases proportionally (v ∝ 1/B).
    # 2. SGD requires linear scaling (η ∝ B) because it directly applies the unnormalized gradient.
    # 3. Adam-family optimizers, however, normalize the update by dividing the 1st moment (m)
    #    by the square root of the 2nd moment (v): Update = η * (m / sqrt(v)).
    #    Because the denominator 'sqrt(v)' naturally shrinks as batch size increases, Adam inherently
    #    amplifies its own step size. For example, if B increases by 4x, variance drops to 1/4,
    #    and the denominator drops to 1/2, naturally making the step size 2x larger.
    #    Therefore, to prevent overshooting, we only need to scale the explicit learning rate (η)
    #    by the square root of the batch ratio (√(B/B_ref)), balancing the optimization trajectory.
    #    (Note: this scaling rule is empirically derived and may not hold for all model architectures, 
    #     but it is a standard practice for scaling training hyperparameters.)
    # 4. Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Calculate the appropriate weight decay scaling based on batch size and token horizon.
# 1. We adopt the T_epoch (Weight Decay Timescale) framework from https://arxiv.org/abs/2405.13698.
#    The core theorem states that for optimal generalization, the timescale T_epoch = B / (eta * lambda * D) 
#    must remain constant across different model scales and datasets.
# 2. Mathematical derivation for lambda_new (weight_decay_scaled):
#    Equate T_epoch for both reference (ref) and target (new) models:
#       B_new / (eta_new * lambda_new * D_new) = B_ref / (eta_ref * lambda_ref * D_ref)
#    Substitute the AdamW Square Root Scaling rule for eta_new (eta_new = eta_ref * sqrt(B_new / B_ref)):
#       B_new / (eta_ref * sqrt(B_new / B_ref) * lambda_new * D_new) = B_ref / (eta_ref * lambda_ref * D_ref)
#    Cancel out eta_ref and solve for lambda_new:
#       lambda_new = lambda_ref * sqrt(B_new / B_ref) * (D_ref / D_new)
# 3. Physical meaning of the two scaling factors:
#    - sqrt(B_new / B_ref) [Batch Size Factor]: Larger batches reduce gradient noise, allowing for larger 
#    step sizes (eta). The per-step weight decay must scale up proportionally to balance this increased step size.
#    - (D_ref / D_new) [Token Horizon Factor]: A larger token horizon (D) means more total optimization steps. 
#    If lambda isn't scaled down, the weights will be penalized too many times over the entire training run, leading to underfitting.
# 4. Note: The T_epoch framework strictly studies AdamW. Since Muon also utilizes momentum dynamics, 
#    we assume this scaling theorem provides a mathematically viable approximation for Muon as well.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled
)


        
    
