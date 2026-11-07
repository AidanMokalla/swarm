"""Swarm runner: synchronous rounds of text-only message passing between agents.

Turn 0: source agents answer the seed prompt. Turn t>0: an agent acts iff some
in-neighbor spoke at turn t-1 (so hop distance == earliest turn an agent acts).
Agent i runs on replica i % n_replicas; one worker thread per replica/GPU.
"""
import dataclasses
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch

from agent import generate_reply, load_persona_vector, load_replica
from config import CFG


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

        def run_group(replica_idx):  # sequential within a replica (one GPU)
            return [run_agent(i, turn, messages[i]) for i in acting if i % len(replicas) == replica_idx]

        with ThreadPoolExecutor(len(replicas)) as pool:
            turn_rows = [row for group in pool.map(run_group, range(len(replicas))) for row in group]
        log.extend((turn, row["agent"], row["text"]) for row in turn_rows)
        rows.extend(turn_rows)
        print(f"turn {turn}: agents {sorted(acting)}")
    run_metadata = {
        "persona_trait": cfg.persona_trait, "layer": cfg.layer, "alpha": cfg.alpha,
        "steering_mode": cfg.steering_mode, "n_agents": cfg.n_agents,
        "topology": cfg.topology, "n_turns": cfg.n_turns,
        "seed_prompt": cfg.seed_prompt, "temperature": cfg.temperature,
    }
    return pd.DataFrame(rows).assign(**run_metadata), histories


def main():
    cfg = CFG
    torch.manual_seed(cfg.seed)
    persona_vector = load_persona_vector(cfg.persona_vector_path, cfg.layer)
    steer_vector = {
        "persona": persona_vector,
        "random": random_matched_norm(persona_vector, cfg.seed),
        "none": None,
    }[cfg.steering_mode]
    replicas = [load_replica(cfg, device) for device in cfg.devices]  # sequential: loading isn't thread-safe
    results, histories = run_swarm(cfg, replicas, persona_vector, steer_vector)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / f"{cfg.steering_mode}.csv", index=False)
    (out_dir / f"{cfg.steering_mode}_config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    (out_dir / f"{cfg.steering_mode}_transcripts.json").write_text(json.dumps(histories, indent=2))
    print(results.drop(columns=["system_prompt", "seed_prompt", "received", "text"]).to_string())


if __name__ == "__main__":
    main()
