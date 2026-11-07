import urllib.request
from pathlib import Path

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


def download_trait_data():
    path = Path(CFG.trait_data_path)
    if path.exists():
        print(f"{path} already present")
        return
    url = CFG.trait_data_url.format(trait=CFG.persona_trait)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, path)
    except OSError as err:
        raise SystemExit(f"failed to fetch {url}: {err}") from err
    print(f"downloaded {path}")


def check_persona_vector():
    # safety-research/persona_vectors does not ship precomputed vectors (gitignored), so nothing to fetch.
    path = Path(CFG.persona_vector_path)
    if path.exists():
        print(f"{path} already present")
        return
    raise SystemExit(
        f"missing {path}: the official repo ships no precomputed vectors. Generate with its "
        f"eval.eval_persona (pos/neg, --version extract, --trait {CFG.persona_trait}) + "
        f"generate_vec.py --model_name {CFG.model_id} --save_dir {CFG.vectors_dir}, "
        f"or build one with agent.extract_persona_vector. Trait data: {CFG.trait_data_path}"
    )


def main():
    download_model()
    download_trait_data()
    check_persona_vector()


if __name__ == "__main__":
    main()
