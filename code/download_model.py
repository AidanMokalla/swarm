"""Fetch the model and generate persona vectors with the official repo (git submodule).

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


def ensure_submodule():
    repo = Path(CFG.persona_repo_dir)
    if not (repo / "generate_vec.py").exists():
        subprocess.run(["git", "submodule", "update", "--init", "--", str(repo)], check=True)
    print(f"persona repo ready at {repo}")


def vector_path(trait):
    return Path(CFG.vectors_dir) / f"{trait}_{CFG.vector_kind}.pt"


def read_env_file(path):
    """KEY=VALUE lines, parsed like the official loader (which overrides os.environ)."""
    if not path.exists():
        return {}
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.lstrip().startswith("#"))
    return {key.strip(): value.strip().strip("\"'") for key, value in pairs}


def check_credentials():
    env = {**os.environ, **read_env_file(Path(CFG.persona_repo_dir) / ".env")}
    missing = [key for key in REQUIRED_KEYS if not env.get(key) or env[key].startswith("your_")]
    if missing:
        sys.exit(f"missing {missing}: export them or put them in {CFG.persona_repo_dir}/.env "
                 "(OpenAI judge scores responses; vectors are built only from high-scoring ones)")


def download_model():
    try:
        path = snapshot_download(CFG.model_id, cache_dir=CFG.cache_dir, local_files_only=True)
        print(f"{CFG.model_id} already cached at {path}")
    except LocalEntryNotFoundError:
        print(f"{CFG.model_id} not cached, downloading (~15GB)...")
        path = snapshot_download(CFG.model_id, cache_dir=CFG.cache_dir)
        print(f"done: {path}")


def run_in_repo(args):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": CFG.vector_gpu}
    if CFG.cache_dir:  # hub cache dir; absolute since cwd changes
        env["HF_HUB_CACHE"] = str(Path(CFG.cache_dir).resolve())
    subprocess.run([sys.executable, *args], cwd=CFG.persona_repo_dir, env=env, check=True)


def generate_vector(trait):
    """Official README pipeline: pos eval, neg eval (judge-scored), then generate_vec.py."""
    model_name = CFG.model_id.split("/")[-1]
    csv_paths = {kind: f"eval_persona_extract/{model_name}/{trait}_{kind}_instruct.csv" for kind in ("pos", "neg")}
    assistant_names = {"pos": trait, "neg": CFG.neg_assistant_name}
    for kind in ("pos", "neg"):  # eval_persona skips a csv that already exists
        run_in_repo([
            "-m", "eval.eval_persona", "--model", CFG.model_id, "--trait", trait,
            "--output_path", csv_paths[kind], "--persona_instruction_type", kind,
            "--assistant_name", assistant_names[kind], "--judge_model", CFG.judge_model,
            "--version", "extract",
        ])
    save_dir = f"persona_vectors/{model_name}"
    run_in_repo([
        "generate_vec.py", "--model_name", CFG.model_id, "--pos_path", csv_paths["pos"],
        "--neg_path", csv_paths["neg"], "--trait", trait, "--save_dir", save_dir,
    ])
    return Path(CFG.persona_repo_dir) / save_dir / f"{trait}_{CFG.vector_kind}.pt"


def install_vector(trait, source):
    target = vector_path(trait)
    tmp_path = target.with_suffix(".part")  # no half-written files
    shutil.copyfile(source, tmp_path)
    tmp_path.rename(target)
    print(f"{target} written")


def main():
    ensure_submodule()
    Path(CFG.vectors_dir).mkdir(parents=True, exist_ok=True)
    missing = [trait for trait in CFG.vector_traits if not vector_path(trait).exists()]
    if missing:
        check_credentials()
    download_model()
    for trait in missing:
        install_vector(trait, generate_vector(trait))
    for trait in CFG.vector_traits:
        print(f"{vector_path(trait)}: {'ok' if vector_path(trait).exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
