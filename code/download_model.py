"""Fetch the Qwen model and the pre-extracted persona vectors (HF dataset CFG.persona_dataset).

Run from code/: python download_model.py
"""
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoConfig

from config import CFG


def model_cached():
    try:
        snapshot_download(CFG.model_id, cache_dir=CFG.cache_dir, local_files_only=True)
        return True
    except LocalEntryNotFoundError:
        return False


def download_model():
    if model_cached():
        return print(f"{CFG.model_id} already cached")
    print(f"{CFG.model_id} not cached, downloading (~15GB)...")
    print(f"done: {snapshot_download(CFG.model_id, cache_dir=CFG.cache_dir)}")


def download_persona_vectors():
    model_config = AutoConfig.from_pretrained(CFG.model_id, cache_dir=CFG.cache_dir)  # no weights
    expected = (model_config.num_hidden_layers + 1, model_config.hidden_size)  # hidden_states rows
    repo_dir = f"vectors/{CFG.model_id.split('/')[-1]}"
    Path(CFG.vectors_dir).mkdir(parents=True, exist_ok=True)
    for trait in CFG.persona_traits:
        name = f"{trait}_{CFG.vector_kind}.pt"
        target = Path(CFG.vectors_dir) / name
        if target.exists():
            print(f"{target} already present")
        else:
            cached = hf_hub_download(CFG.persona_dataset, f"{repo_dir}/{name}", repo_type="dataset",
                                     cache_dir=CFG.cache_dir)
            shutil.copyfile(cached, target)
        shape = tuple(torch.load(target, map_location="cpu").shape)
        assert shape == expected, f"{target}: shape {shape}, expected {expected}"
        print(f"{target}: {shape}")


if __name__ == "__main__":
    download_model()
    download_persona_vectors()
