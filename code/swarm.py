"""Swarm runner: synchronous rounds of text-only message passing between agents.

Turn 0: source agents answer the seed prompt. Turn t>0: an agent acts iff some
in-neighbor spoke at turn t-1 (so hop distance == earliest turn an agent acts).
Agent i runs on replica i % n_replicas; one worker thread per replica/GPU.
"""
import dataclasses
import hashlib
import itertools
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch

from agent import generate_reply, load_persona_vector, load_replica
from config import CFG, REPLICA_FIELDS, Config


def build_edges(cfg):
    """Directed (src, dst) edge list for cfg.topology."""
    n = cfg.n_agents
    if cfg.topology == "custom":
        return list(cfg.custom_edges)
    if cfg.topology == "chain":
        return [(i, i + 1) for i in range(n - 1)]
    if cfg.topology == "ring":
        return [(i, (i + 1) % n) for i in range(n)]
    if cfg.topology == "star":  # hub = agent 0, bidirectional
        return [(0, i) for i in range(1, n)] + [(i, 0) for i in range(1, n)]
    if cfg.topology == "full":
        return [(i, j) for i in range(n) for j in range(n) if i != j]
    raise ValueError(f"unknown topology {cfg.topology!r}")


def hop_distances(n_agents, edges, sources):
    """BFS hops from nearest source along directed edges; -1 if unreachable."""
    hops = {i: -1 for i in range(n_agents)}
    queue = deque(sources)
    hops.update({s: 0 for s in sources})
    while queue:
        node = queue.popleft()
        for src, dst in edges:
            if src == node and hops[dst] < 0:
                hops[dst] = hops[node] + 1
                queue.append(dst)
    return hops


def random_matched_norm(vector, seed):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(vector.shape, generator=generator)
    return noise * vector.norm() / noise.norm()


def run_swarm(cfg, replicas, persona_vector, steer_vector):
    n = cfg.n_agents
    edges = build_edges(cfg)
    in_neighbors = {i: {s for s, d in edges if d == i} for i in range(n)}
    hops = hop_distances(n, edges, cfg.source_agents)
    histories = {
        i: [{"role": "system", "content": cfg.system_prompts[i % len(cfg.system_prompts)]}]
        for i in range(n)
    }
    log, seen, rows = [], [0] * n, []  # log: (turn, agent, text)

    def user_message(i, turn):
        if turn == 0:
            return cfg.seed_prompt if i in cfg.source_agents else None
        latest = [(a, t) for tn, a, t in log if tn == turn - 1 and a in in_neighbors[i]]
        if not latest:
            return None
        # isolated: only in-neighbors' latest msgs; else: everything unseen from anyone
        heard = latest if cfg.hop_isolated else [(a, t) for _, a, t in log[seen[i]:] if a != i]
        return "\n\n".join(f"[agent {a}]: {t}" for a, t in heard)

    def run_agent(i, turn, message):
        histories[i].append({"role": "user", "content": message})
        steered = steer_vector is not None and i in cfg.source_agents
        text, measurement = generate_reply(
            replicas[i % len(replicas)], histories[i], cfg,
            steer_vector if steered else None, persona_vector,
        )
        histories[i].append({"role": "assistant", "content": text})
        print(  # single call so threads on different GPUs don't interleave mid-message
            f"\n--- turn {turn} | agent {i} | hop {hops[i]}{' | STEERED' if steered else ''}"
            f" | prompt_proj {measurement['prompt_proj']:+.2f} gen_proj {measurement['gen_proj']:+.2f}\n"
            f"<< received:\n{message}\n>> reply:\n{text}",
            flush=True,
        )
        return {"turn": turn, "agent": i, "hop": hops[i], "steered": steered,
                **measurement, "system_prompt": histories[i][0]["content"],
                "received": message, "text": text}

    for turn in range(cfg.n_turns):
        messages = {i: user_message(i, turn) for i in range(n)}
        for i in range(n):
            seen[i] = len(log)
        acting = [i for i in range(n) if messages[i] is not None]
        if not acting:
            break

        print(f"\n===== turn {turn}: agents {sorted(acting)} =====", flush=True)

        def run_group(replica_idx):  # sequential within a replica (one GPU)
            return [run_agent(i, turn, messages[i]) for i in acting if i % len(replicas) == replica_idx]

        with ThreadPoolExecutor(len(replicas)) as pool:
            turn_rows = [row for group in pool.map(run_group, range(len(replicas))) for row in group]
        log.extend((turn, row["agent"], row["text"]) for row in turn_rows)
        rows.extend(turn_rows)
    run_metadata = {
        "persona_trait": cfg.persona_trait, "layer": cfg.layer, "alpha": cfg.alpha,
        "steering_mode": cfg.steering_mode, "n_agents": cfg.n_agents,
        "topology": cfg.topology, "n_turns": cfg.n_turns,
        "seed_prompt": cfg.seed_prompt, "temperature": cfg.temperature, "seed": cfg.seed,
        "hop_isolated": cfg.hop_isolated, "source_agents": str(cfg.source_agents),
    }
    return pd.DataFrame(rows).assign(**run_metadata), histories


def expand_sweep(cfg):
    """One resolved Config (sweep=()) per cartesian-product combo of cfg.sweep.

    Mode 'none' ignores alpha, so alpha is reset to the base value for it and duplicate
    configs are dropped (order kept). It still runs per persona_trait/layer: those set
    the measurement vector.
    """
    names = [name for name, _ in cfg.sweep]
    sweepable = {f.name for f in dataclasses.fields(Config)} - {"sweep", *REPLICA_FIELDS}
    if bad := [n for n in names if n not in sweepable]:
        raise ValueError(f"cannot sweep {bad}; sweepable fields: {sorted(sweepable)}")
    if len(set(names)) < len(names) or not all(values for _, values in cfg.sweep):
        raise ValueError(f"sweep needs unique names and non-empty values: {cfg.sweep}")
    base = dataclasses.replace(cfg, sweep=())
    runs = []
    for values in itertools.product(*(values for _, values in cfg.sweep)):
        run = dataclasses.replace(base, **dict(zip(names, values)))
        runs.append(dataclasses.replace(run, alpha=base.alpha) if run.steering_mode == "none" else run)
    return list(dict.fromkeys(runs))


def run_name(cfg):
    """Readable filename stem + short hash of the full resolved config (unique, deterministic)."""
    digest = hashlib.sha1(json.dumps(dataclasses.asdict(cfg), sort_keys=True).encode()).hexdigest()[:6]
    return (f"{cfg.steering_mode}_{cfg.persona_trait}_L{cfg.layer}_a{cfg.alpha:g}_{cfg.topology}"
            f"_n{cfg.n_agents}_t{cfg.n_turns}_T{cfg.temperature:g}_s{cfg.seed}_{digest}")


def main():
    runs = expand_sweep(CFG)
    replicas = None  # loaded on first run that needs it; shared by all runs
    for k, cfg in enumerate(runs, 1):
        name, out_dir = run_name(cfg), Path(cfg.output_dir)
        print(f"[{k}/{len(runs)}] {name}")
        if (out_dir / f"{name}.csv").exists():
            print("  exists, skipping")
            continue
        if replicas is None:  # sequential: loading isn't thread-safe
            replicas = [load_replica(CFG, device) for device in CFG.devices]
        torch.manual_seed(cfg.seed)
        persona_vector = load_persona_vector(cfg.persona_vector_path, cfg.layer)
        steer_vector = {
            "persona": persona_vector,
            "random": random_matched_norm(persona_vector, cfg.seed),
            "none": None,
        }[cfg.steering_mode]
        results, histories = run_swarm(cfg, replicas, persona_vector, steer_vector)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}_config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
        (out_dir / f"{name}_transcripts.json").write_text(json.dumps(histories, indent=2))
        results.to_csv(out_dir / f"{name}.csv", index=False)  # last: its existence marks the run done
        print(results.drop(columns=["system_prompt", "seed_prompt", "received", "text"]).to_string())


if __name__ == "__main__":
    main()
