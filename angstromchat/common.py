import angstromchat
import os

def get_base_dir():
    if os.environ.get("ANGSTROMCHAT_BASE_DIR"):
        angstromchat_dir = os.environ.get("ANGSTROMCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        angstromchat_dir = os.path.join(cache_dir, "angstromchat")
    os.makedirs(angstromchat_dir, exist_ok=True)
    return angstromchat_dir

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