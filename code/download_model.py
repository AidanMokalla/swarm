from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from config import CFG


def download_model():
    try:
        path = snapshot_download(CFG.model_id, cache_dir=CFG.cache_dir, local_files_only=True)
        print(f"{CFG.model_id} already cached at {path}")
    except LocalEntryNotFoundError:
        print(f"{CFG.model_id} not cached, downloading (~15GB)...")
        path = snapshot_download(CFG.model_id, cache_dir=CFG.cache_dir)
        print(f"done: {path}")


def download_vectors():
    out_dir = Path(CFG.vectors_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for trait in CFG.vector_traits:
        name = f"{trait}_{CFG.vector_kind}.pt"
        path = out_dir / name
        if path.exists():
            print(f"{path} already present")
            continue
        tmp_path = path.with_suffix(".part")  # no half-written files
        urlretrieve(f"{CFG.vectors_url}/{name}", tmp_path)
        tmp_path.rename(path)
        print(f"{path} downloaded")


def main():
    download_model()
    download_vectors()


if __name__ == "__main__":
    main()
