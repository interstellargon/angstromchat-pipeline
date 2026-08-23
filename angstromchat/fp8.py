"""Minimal FP8 training for angstromchat — tensorwise dynamic scaling only.

How FP8 training works
======================
A standard Linear layer does one matmul in forward and two in backward:
  forward:      output     = input      @ weight.T
  backward:     grad_input = grad_output @ weight
                grad_weight= grad_output.T @ input

FP8 training wraps each of these three matmuls with:
  1. Compute scale = FP8_MAX / max(|tensor|)  for each operand
  2. Quantize: fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
  3. Matmul via torch._scaled_mm (cuBLAS FP8 kernel, ~2x faster than bf16)
  4. Dequantize: _scaled_mm handles this internally using the inverse scales

FP8 dtype choice
================
There are two FP8 formats. We use both, following the standard convention:
  - float8_e4m3fn: 4-bit exponent, 3-bit mantissa, range [-448, 448]
    Higher precision (more mantissa bits), used for input and weight.
  - float8_e5m2:   5-bit exponent, 2-bit mantissa, range [-57344, 57344]
    Wider range (more exponent bits), used for gradients which can be large.

torch._scaled_mm layout requirements
=====================================
The cuBLAS FP8 kernel requires specific memory layouts:
  - First argument (A):  must be row-major (contiguous)
  - Second argument (B): must be column-major (B.t().contiguous().t())
If B is obtained by transposing a contiguous tensor (e.g. weight.t()), it is
already column-major — no copy needed. Otherwise we use _to_col_major().
"""

import torch
import torch.nn as nn


class Float8LinearConfig:
  """Minimal config matching torchao's API. Currently only support tensorwise recipe"""
  @staticmethod
  def from_recipe_name(recipe_name):
    if recipe_name != "tensorwise":
      raise ValueError(
          f"Only 'tensorwise' recipe is supported, got '{recipe_name}'. "
          f"Rowwise/axiswise recipes require the full torchao library."
      )
    return Float8LinearConfig()


class Float8Linear(nn.Linear):
    """Drop-in nn.Linear replacement that does FP8 compute.

    Weights and biases remain in their original precision (e.g. fp32/bf16).
    Only the matmul is performed in FP8 via the _Float8Matmul autograd function.
    """

    @classmethod
    def from_float(cls, mod):
        """Create Float8Linear from nn.Linear, sharing the same weight and bias.
        Uses meta device to avoid allocating a temporary weight tensor.
        """
        with torch.device("meta"):
          new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
    """Replace nn.Linear layers with Float8Linear throughout a module.

    Walks the module tree in post-order (children before parents) and swaps
    each nn.Linear that passes the optional filter. The new Float8Linear shares
    the original weight and bias tensors — no copies, no extra memory.

    Args:
        module: Root module to convert.
        config: Float8LinearConfig (accepted for API compat, only tensorwise supported).
        module_filter_fn: Optional filter(module, fqn) -> bool. Only matching Linears
            are converted. Common use: skip layers with dims not divisible by 16
            (hardware requirement for FP8 matmuls on H100).
    """
    def _convert(mod, prefix=""):
      for name, child in mod.named_children():
        fqn = f"{prefix}.{name}" if prefix else name
        _convert(child, fqn)
        if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
          if module_filter_fn is None or module_filter_fn(child, fqn):
            setattr(mod, name, Float8Linear.from_float(child))          

    _convert(module)
    return module


