"""
Common utilities for angstromchat.
"""

import logging
import os
from filelock import FileLock
import urllib.request

import torch
import torch.distributed as dist


def setup_default_logging():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.basicConfig(
        level = logging.INFO,
        handlers= [handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)
        
def get_base_dir():
    if os.environ.get("ANGSTROMCHAT_BASE_DIR"):
        angstromchat_dir = os.environ.get("ANGSTROMCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        angstromchat_dir = os.path.join(cache_dir, "angstromchat")
    os.makedirs(angstromchat_dir, exist_ok=True)
    return angstromchat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()  # bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def is_ddp_requested() -> bool:
    return all(var in os.environ for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def get_dist_info():
    if is_ddp_requested():
        assert all(var in os.environ for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])   
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    if torch.cuda.is_available():
        device_type = "cuda"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="cuda"):
    assert device_type in ["cuda", "cpu"], "Input invalid device type"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your pytorch installation does not have cuda support, but you requested cuda"

    # Reproducibility
    # Set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)

    # Precision
    if device_type == "cuda":
        torch.backends.fp32_precision = "tf32"  # uses tf32 instead of fp32 for matmuls.

    # Distributed setup
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")        

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device


# hardcoded BF16 peak flops for various GPUs
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX
        (["5090"], 209.5e12),
        (["5080"], 150.0e12),   
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')


class DummyWandb:
    """Useful if do not wish to use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass
