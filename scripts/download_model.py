from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-27B",
    local_dir="./models/qwen3.5-27b",
    local_dir_use_symlinks=False
)
