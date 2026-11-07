from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from config import CFG


def main():
    try:
        path = snapshot_download(CFG.model_id, cache_dir=CFG.cache_dir, local_files_only=True)
        print(f"{CFG.model_id} already cached at {path}")
    except LocalEntryNotFoundError:
        print(f"{CFG.model_id} not cached, downloading (~15GB)...")
        path = snapshot_download(CFG.model_id, cache_dir=CFG.cache_dir)
        print(f"done: {path}")


if __name__ == "__main__":
    main()
