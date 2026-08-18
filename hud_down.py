import os
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "youliangtan/bridge_dataset"
LOCAL_DIR = r"D:\code\llm_new\vla\bridge_dataset"
MAX_SHARDS = 52  # 需要下载的 TFRecord 分片数量

os.makedirs(LOCAL_DIR, exist_ok=True)
api = HfApi()

print(f"正在扫描 Hugging Face 仓库: {REPO_ID} ...")
# 扫描 1.0.0 目录下的所有文件
all_files = [
    f.path for f in api.list_repo_tree(repo_id=REPO_ID, repo_type="dataset", path_in_repo="1.0.0")
]

# 严格过滤：仅保留基础 .json 元数据与 .tfrecord 分片，彻底剔除非法 html 网页文件
meta_files = [f for f in all_files if f.endswith(".json")]
shard_files = sorted([f for f in all_files if "tfrecord" in f])[:MAX_SHARDS]

targets = meta_files + shard_files
print(f"筛选出核心元数据 {len(meta_files)} 个，TFRecord 分片 {len(shard_files)} 个。")

for remote_path in targets:
    filename = os.path.basename(remote_path)
    target_path = os.path.join(LOCAL_DIR, filename)

    # 跳过已完整下载的文件
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        print(f"⏭️  [已存在跳过] {filename}")
        continue

    print(f"⬇️  正在下载: {filename} ...")
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=remote_path,
            local_dir=LOCAL_DIR,
        )
        print(f"✅ [完成] {filename}")
    except Exception as e:
        print(f"⚠️ [跳过错误文件] {filename}: {e}")

print(f"\n🎉 目标 {len(shard_files)} 个分片及元数据全部同步完毕！存放路径: {LOCAL_DIR}")