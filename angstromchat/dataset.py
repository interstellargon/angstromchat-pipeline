import argparse
import os
from multiprocessing import Pool
import requests
import time
import pyarrow.parquet as pq

from angstromchat.common import get_base_dir


# The URL where the dataset is hosted
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
BASE_DIR = get_base_dir()
DATA_DIR = os.path.join(BASE_DIR, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

#--------------------------------------------------------------------------------------------
# These functions are used to other modules
def list_parquet_files(data_dir=None):
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])    
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Memory-efficient batch generator for Parquet files, optimized for DDP.

    - Train/Val Split: Automatically reserves the last file in the directory for validation and the rest for training.
    - Memory Efficiency: Reads data strictly at the row-group level using PyArrow to prevent OOM errors.
    - Multi-GPU Support: Uses `start` (rank) and `step` (world_size) to yield non-overlapping batches across devices.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == 'train' else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

#--------------------------------------------------------------------------------------------
def download_single_file(index):
    # Construct the local filepath for this file
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} as it already exists.")
        return True     
    
    # Download the file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts+1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temppath = filepath + f".tmp"
            with open(temppath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temppath, filepath)
            print(f"Successfully downloads {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts.")
                return False

    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset")
    parser.add_argument("-n", "--num-shards", type=int, default=-1, help="Number of shards to download, -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers")
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_shards == -1 else min(args.num_shards, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers.")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Downloading is completed, {successful}/{len(ids_to_download)} shards downloaded to {DATA_DIR}")

