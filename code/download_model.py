"""Fetch the model and generate persona vectors with the official repo (vendored in CFG.persona_repo_dir).

Run from code/. Needs a GPU, OPENAI_API_KEY and HF_TOKEN (env or <persona_repo_dir>/.env).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from config import CFG

REQUIRED_KEYS = ("OPENAI_API_KEY", "HF_TOKEN")  # the repo's config.py demands both at import
MODEL_NAME = CFG.model_id.split("/")[-1]


def vector_path(trait):
    return Path(CFG.vectors_dir) / f"{trait}_{CFG.vector_kind}.pt"


def read_env_file(path):
    """KEY=VALUE lines, parsed like the official loader (which overrides os.environ)."""
    lines = path.read_text().splitlines() if path.exists() else []
    pairs = (line.split("=", 1) for line in lines if "=" in line and not line.lstrip().startswith("#"))
    return {key.strip(): value.strip().strip("\"'") for key, value in pairs}


def check_credentials():
    env = {**os.environ, **read_env_file(Path(CFG.persona_repo_dir) / ".env")}
    missing = [key for key in REQUIRED_KEYS if not env.get(key) or env[key].startswith("your_")]
    if missing:
        sys.exit(f"missing {missing}: export them or put them in {CFG.persona_repo_dir}/.env "
                 "(the OpenAI judge scores responses; vectors use only high-scoring ones)")


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


def vector_commands(trait):
    """README pipeline (scripts/generate_vec.sh): pos eval, neg eval, generate_vec.py. Paths are relative to the repo dir."""
    csv = {kind: f"eval_persona_extract/{MODEL_NAME}/{trait}_{kind}_instruct.csv" for kind in ("pos", "neg")}
    names = {"pos": trait, "neg": CFG.neg_assistant_name}
    evals = [["-m", "eval.eval_persona", "--model", CFG.model_id, "--trait", trait, "--output_path", csv[kind],
              "--persona_instruction_type", kind, "--assistant_name", names[kind],
              "--judge_model", CFG.judge_model, "--version", "extract"] for kind in ("pos", "neg")]  # skips existing csv
    save_dir = f"persona_vectors/{MODEL_NAME}"
    generate = ["generate_vec.py", "--model_name", CFG.model_id, "--pos_path", csv["pos"], "--neg_path", csv["neg"],
                "--trait", trait, "--save_dir", save_dir, "--threshold", str(CFG.vector_threshold)]
    return [*evals, generate], Path(CFG.persona_repo_dir) / save_dir / f"{trait}_{CFG.vector_kind}.pt"


def generate_persona_vectors():
    missing = [trait for trait in CFG.vector_traits if not vector_path(trait).exists()]
    for trait in set(CFG.vector_traits) - set(missing):
        print(f"{vector_path(trait)} already present")
    if not missing:
        return
    check_credentials()  # before spending GPU time
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": CFG.vector_gpu}
    if CFG.cache_dir:  # absolute since cwd changes
        env["HF_HUB_CACHE"] = str(Path(CFG.cache_dir).resolve())
    for trait in missing:
        commands, generated = vector_commands(trait)
        for args in commands:
            subprocess.run([sys.executable, *args], cwd=CFG.persona_repo_dir, env=env, check=True)
        target = vector_path(trait)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated, target)
        print(f"{target} written")


def main():
    assert (Path(CFG.persona_repo_dir) / "generate_vec.py").exists(), f"{CFG.persona_repo_dir}/ missing (vendored repo)"
    download_model()
    generate_persona_vectors()


if __name__ == "__main__":
    main()
