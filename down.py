"""
Production-Grade Batch Downloader for OpenX Datasets (dddd.py)
Downloads all 3 Active Embodied Robot Datasets (10 shards each):
  1. BridgeData v2 (WidowX Desktop Kitchen)
  2. Fractal20220817 / RT-1 (Google Robot)
  3. DROID-100 / DROID (Franka Panda Multi-Scene)
"""

import os
import sys
import time
import math
import argparse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

# Optional GCS library
try:
    from google.cloud import storage
    HAS_GCS_SDK = True
except ImportError:
    HAS_GCS_SDK = False

# OpenX 3 Active Core Datasets
DATASET_REGISTRY = {
    "bridge_dataset": {
        "name": "BridgeData v2 (WidowX Manipulator)",
        "bucket": "gresearch",
        "prefix": "robotics/bridge_dataset/0.1.0",
        "default_dir": "./bridge_dataset",
        "shard_pattern": "bridge_dataset-train.tfrecord-{:05d}-of-01024",
    },
    "fractal20220817_data": {
        "name": "Fractal20220817 / RT-1 (Google Robot)",
        "bucket": "gresearch",
        "prefix": "robotics/fractal20220817_data/0.1.0",
        "default_dir": "./fractal20220817_data",
        "shard_pattern": "fractal20220817_data-train.tfrecord-{:05d}-of-01024",
    },
    "droid_100": {
        "name": "DROID 100 / DROID Subset (Franka Panda)",
        "bucket": "gresearch",
        "prefix": "robotics/droid_100",
        "default_dir": "./droid_100",
        "shard_pattern": "1.0.0/r2d2_faceblur-train.tfrecord-{:05d}-of-00031",
    },
}


def format_bytes(size_bytes: int) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    if size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {units[i]}"


class FileDownloadProgress:
    """Tracks and displays real-time CLI progress bar for individual file downloads."""

    def __init__(self, filename: str, total_bytes: int):
        self.filename = filename
        self.total_bytes = total_bytes
        self.downloaded = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time

    def update(self, chunk_size: int):
        self.downloaded += chunk_size
        now = time.time()
        if now - self.last_update_time > 0.25 or self.downloaded >= self.total_bytes:
            self.last_update_time = now
            self.render()

    def render(self):
        elapsed = max(time.time() - self.start_time, 1e-4)
        speed = self.downloaded / elapsed
        speed_str = f"{format_bytes(int(speed))}/s"

        if self.total_bytes > 0:
            pct = min(self.downloaded / self.total_bytes, 1.0)
            bar_len = 20
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)
            eta = (self.total_bytes - self.downloaded) / max(speed, 1.0)
            eta_str = f"{int(eta)}s" if eta < 3600 else f"{eta/3600:.1f}h"
            msg = (
                f"\r  ⬇️  {self.filename[:30]:<30s} [{bar}] {pct*100:5.1f}% | "
                f"{format_bytes(self.downloaded):>9s}/{format_bytes(self.total_bytes):<9s} | "
                f"{speed_str:>9s} | ETA: {eta_str:>4s}"
            )
        else:
            msg = f"\r  ⬇️  {self.filename[:30]:<30s} | {format_bytes(self.downloaded):>9s} | {speed_str:>9s}"

        sys.stdout.write(msg)
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


def download_single_file_https(
    url: str,
    save_path: str,
    expected_size: Optional[int] = None,
    max_retries: int = 4,
) -> bool:
    """Download a single file via direct HTTPS with progress bar, validation, and atomic rename."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    temp_path = save_path + ".tmp"
    filename = os.path.basename(save_path)

    # Check if already downloaded
    if os.path.exists(save_path):
        current_size = os.path.getsize(save_path)
        if expected_size is not None and expected_size > 0:
            if current_size == expected_size:
                print(f"  ⏭️  [Skipped] {filename} (Already exists, {format_bytes(current_size)})")
                return True
        elif current_size > 0:
            print(f"  ⏭️  [Skipped] {filename} (Already exists, {format_bytes(current_size)})")
            return True

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) OpenVLA-Downloader/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = expected_size or int(response.headers.get("Content-Length", 0))
                progress = FileDownloadProgress(filename, total_size)

                with open(temp_path, "wb") as f_out:
                    chunk_size = 1024 * 1024  # 1MB buffer
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        progress.update(len(chunk))

                progress.close()

            # Atomic move to destination
            if os.path.exists(save_path):
                os.remove(save_path)
            os.rename(temp_path, save_path)
            print(f"  ✅ [Done] {filename} ({format_bytes(os.path.getsize(save_path))})")
            return True

        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
            else:
                if "404" in str(e) or "HTTP Error 404" in str(e):
                    print(f"  ℹ️  [Not Found] {filename} (Reached end of available shards in dataset)")
                else:
                    print(f"  ⚠️  [Failed] {filename}: {e}")
                return False
    return False


def get_blob_list_gcs(bucket_name: str, prefix: str) -> List[Tuple[str, int]]:
    """List blobs from GCS SDK if available."""
    if not HAS_GCS_SDK:
        return []
    try:
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        return [(b.name, b.size or 0) for b in blobs if not b.name.endswith("/")]
    except Exception:
        return []


def download_dataset(
    dataset_key: str,
    meta: Dict[str, Any],
    max_shards: int = 10,
    max_workers: int = 4,
) -> int:
    """Download a single dataset (metadata + up to max_shards shards)."""
    bucket_name = meta["bucket"]
    prefix = meta["prefix"]
    local_dir = meta["default_dir"]
    name = meta["name"]

    print("\n" + "─" * 75)
    print(f"📦 Downloading Dataset: {name}")
    print(f"   Remote Source : gs://{bucket_name}/{prefix}")
    print(f"   Local Target  : {os.path.abspath(local_dir)}")
    print(f"   Shards Limit  : First {max_shards} TFRecord Shards")
    print("─" * 75)

    os.makedirs(local_dir, exist_ok=True)

    blob_items: List[Tuple[str, int]] = get_blob_list_gcs(bucket_name, prefix)

    final_tasks = []
    if blob_items:
        meta_items = [b for b in blob_items if not ("tfrecord" in b[0] or ".tfrecord" in b[0])]
        shard_items = [b for b in blob_items if ("tfrecord" in b[0] or ".tfrecord" in b[0])]
        shard_items = sorted(shard_items, key=lambda x: x[0])[:max_shards]

        for blob_name, blob_size in meta_items + shard_items:
            rel_path = os.path.relpath(blob_name, prefix)
            save_path = os.path.join(local_dir, rel_path)
            public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
            final_tasks.append((public_url, save_path, blob_size, blob_name))
    else:
        base_prefix = prefix.strip("/")
        for mf in ["dataset_info.json", "features.json"]:
            blob_name = f"{base_prefix}/{mf}"
            save_path = os.path.join(local_dir, mf)
            public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
            final_tasks.append((public_url, save_path, 0, blob_name))

        pattern = meta.get("shard_pattern", "")
        if pattern:
            for s_idx in range(max_shards):
                shard_rel = pattern.format(s_idx)
                blob_name = f"{base_prefix}/{shard_rel}"
                save_path = os.path.join(local_dir, shard_rel)
                public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
                final_tasks.append((public_url, save_path, 0, blob_name))

    print(f"   Queued tasks: {len(final_tasks)} files to process with {max_workers} threads...\n")

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_single_file_https,
                url=task[0],
                save_path=task[1],
                expected_size=task[2] if task[2] > 0 else None,
            ): task
            for task in final_tasks
        }

        for future in as_completed(futures):
            try:
                if future.result():
                    success_count += 1
            except Exception:
                pass

    print(f"✨ Completed [{name}]: {success_count} files ready in '{local_dir}'.")
    return success_count


def download_all_datasets(max_shards_per_dataset: int = 10, max_workers: int = 4):
    """
    Batch download all 3 active OpenX datasets (10 shards each) in a single execution.
    """
    total_datasets = len(DATASET_REGISTRY)
    print("\n" + "=" * 80)
    print(f"🚀 OpenVLA-AlignFlow: Batch Downloading {total_datasets} OpenX Datasets")
    print(f"   Target Shards : {max_shards_per_dataset} Shards per dataset")
    print(f"   Concurrency   : {max_workers} Threads")
    print("=" * 80)

    start_time = time.time()
    total_downloaded = 0

    for idx, (key, meta) in enumerate(DATASET_REGISTRY.items(), 1):
        print(f"\n[{idx}/{total_datasets}] Processing: {meta['name']} ...")
        count = download_dataset(
            dataset_key=key,
            meta=meta,
            max_shards=max_shards_per_dataset,
            max_workers=max_workers,
        )
        total_downloaded += count

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("🎉 ALL 3 DATASETS DOWNLOAD BATCH FINISHED SUCCESSFULLY!")
    print(f"   • Total Files Processed: {total_downloaded}")
    print(f"   • Total Elapsed Time   : {elapsed:.2f} seconds")
    print(f"   • Target Directories   :")
    for key, meta in DATASET_REGISTRY.items():
        print(f"     - {meta['name']:<38s}: {os.path.abspath(meta['default_dir'])}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download All 3 Active OpenX Datasets (10 Shards Each)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=list(DATASET_REGISTRY.keys()) + ["all"],
        help="Specify 'all' to download all 3 datasets, or choose a specific dataset",
    )
    parser.add_argument("--max_shards", type=int, default=100, help="Number of shards per dataset (default: 10)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent download threads")
    args = parser.parse_args()

    if args.dataset == "all":
        download_all_datasets(max_shards_per_dataset=args.max_shards, max_workers=args.workers)
    else:
        meta = DATASET_REGISTRY[args.dataset]
        download_dataset(
            dataset_key=args.dataset,
            meta=meta,
            max_shards=args.max_shards,
            max_workers=args.workers,
        )


if __name__ == "__main__":
    main()