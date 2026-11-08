"""Swarm runner: synchronous waves of text-only dialogue between agents.

Turn 0: source agents answer the seed prompt. Turn t>0: an agent acts iff some
in-neighbor spoke at turn t-1 (so hop distance == earliest turn an agent acts); each edge
into it gets a back-and-forth of cfg.exchanges_per_edge round trips (see run_swarm).
Agent i runs on replica i % n_replicas; one worker thread per replica/GPU.
"""
import dataclasses
import hashlib
import itertools
import json
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch

from agent import generate_reply, load_persona_vector, load_replica, scale_to_norm
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


def random_vector(hidden_size, norm, seed):
    return scale_to_norm(torch.randn(hidden_size, generator=torch.Generator().manual_seed(seed)), norm)


def measured_vector_name(cfg):
    return f"random_s{cfg.seed}" if cfg.steering_mode == "random" else f"persona:{cfg.persona_trait}"


def schedule(pairs):
    """Split (sender, receiver) pairs, sorted by (receiver, sender), into slots of agent-disjoint
    pairs (greedy; each agent's pairs keep their order). Pairs in one slot talk in parallel."""
    free, slots = Counter(), []  # free[agent]: first slot where the agent is available
    for sender, receiver in pairs:
        k = max(free[sender], free[receiver])
        free[sender] = free[receiver] = k + 1
        if k == len(slots):
            slots.append([])
        slots[k].append((sender, receiver))
    return slots


def run_swarm(cfg, replicas, measure_vector, steer_vector):
    """Waves of dialogue. Wave 0 (turn 0): sources answer the seed prompt. Wave t > 0: each edge
    (S, R) whose sender spoke in wave t-1 gets a dialogue of cfg.exchanges_per_edge round trips:
    S's latest message -> R replies -> S replies -> R replies ... (R speaks E times, S E-1).
    R talks to its in-neighbors one after another (by id); its last reply is what its out-neighbors
    see in wave t+1 (replies made as a sender never are). n_turns bounds waves, not messages."""
    if cfg.exchanges_per_edge < 1:
        raise ValueError(f"exchanges_per_edge must be >= 1, got {cfg.exchanges_per_edge}")
    n = cfg.n_agents
    edges = set(build_edges(cfg))
    hops = hop_distances(n, edges, cfg.source_agents)
    histories = {
        i: [{"role": "system", "content": cfg.system_prompts[i % len(cfg.system_prompts)]}]
        for i in range(n)
    }
    n_calls = Counter()  # (agent, turn) -> generate calls so far; seeds the next one (see derive_seed)
    rows, out, pending = [], {}, {}  # out: agent -> message its out-neighbors hear this wave

    def opening(sender, receiver):
        heard = {sender: out[sender]}  # isolated: only the in-neighbor's latest msg
        if not cfg.hop_isolated:  # else: also everything else the receiver hasn't been shown yet
            heard.update(pending[receiver])
            pending[receiver].clear()
        return "\n\n".join(f"[agent {a}]: {t}" for a, t in heard.items())

    def speak(turn, i, partner, exchange, message):
        histories[i].append({"role": "user", "content": message})
        steered = steer_vector is not None and i in cfg.source_agents  # steered on every reply
        text, measurement = generate_reply(
            replicas[i % len(replicas)], histories[i], cfg,
            steer_vector if steered else None, measure_vector,
            agent_id=i, turn=turn, generation=n_calls[i, turn],
        )
        n_calls[i, turn] += 1
        histories[i].append({"role": "assistant", "content": text})
        print(  # single call so threads on different GPUs don't interleave mid-message
            f"\n--- turn {turn} | exchange {exchange} | agent {i} <- {partner} | hop {hops[i]}"
            f"{' | STEERED' if steered else ''}"
            f" | prompt_proj {measurement['prompt_proj']:+.2f} gen_proj {measurement['gen_proj']:+.2f}\n"
            f"<< received:\n{message}\n>> reply:\n{text}",
            flush=True,
        )
        return {"turn": turn, "exchange": exchange, "agent": i, "partner": partner,
                "hop": hops[i], "steered": steered, **measurement,
                "system_prompt": histories[i][0]["content"], "received": message, "text": text}

    def speak_all(turn, jobs):
        """jobs: (agent, partner, exchange, message), one per agent. One thread per replica/GPU,
        sequential within a replica; rows come back in a thread-order-independent order."""
        def run_group(replica_idx):
            return [speak(turn, *job) for job in jobs if job[0] % len(replicas) == replica_idx]

        with ThreadPoolExecutor(len(replicas)) as pool:
            return [row for group in pool.map(run_group, range(len(replicas))) for row in group]

    for turn in range(cfg.n_turns):
        if turn == 0:
            print(f"\n===== turn 0: sources {sorted(cfg.source_agents)} =====", flush=True)
            wave_rows = speak_all(turn, [(i, -1, 0, cfg.seed_prompt) for i in sorted(cfg.source_agents)])
            final = {row["agent"]: row["text"] for row in wave_rows}
        else:
            pairs = sorted(((s, r) for s, r in edges if s in out), key=lambda pair: pair[::-1])
            if not pairs:
                break
            print(f"\n===== turn {turn}: {len(pairs)} edges =====", flush=True)
            pending = {r: {a: t for a, t in out.items() if a != r} for _, r in pairs}
            final, wave_rows = {}, []  # final: receiver -> its last reply this wave
            for slot in schedule(pairs):
                incoming = {pair: opening(*pair) for pair in slot}  # message awaiting each speaker
                pair_of = {a: pair for pair in slot for a in pair}
                for step in range(2 * cfg.exchanges_per_edge - 1):  # R, S, R, S, ..., R
                    jobs = [(*(pair[::-1] if step % 2 == 0 else pair), (step + 1) // 2, incoming[pair])
                            for pair in slot]
                    step_rows = speak_all(turn, jobs)
                    for row in step_rows:
                        incoming[pair_of[row["agent"]]] = f"[agent {row['agent']}]: {row['text']}"
                        if step % 2 == 0:
                            final[row["agent"]] = row["text"]
                    wave_rows.extend(step_rows)
        out = final
        rows.extend(wave_rows)
    run_metadata = {
        "persona_trait": cfg.persona_trait, "layer": cfg.layer, "alpha": cfg.alpha,
        "steer_vector_norm": cfg.steer_vector_norm, "measured_vector": measured_vector_name(cfg),
        "steering_mode": cfg.steering_mode, "n_agents": cfg.n_agents,
        "topology": cfg.topology, "n_turns": cfg.n_turns,
        "exchanges_per_edge": cfg.exchanges_per_edge,
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
            f"_n{cfg.n_agents}_t{cfg.n_turns}_x{cfg.exchanges_per_edge}_T{cfg.temperature:g}_s{cfg.seed}_{digest}")


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
        persona_vector = load_persona_vector(cfg.persona_vector_path, cfg.layer, cfg.steer_vector_norm)
        random_vec = random_vector(persona_vector.shape[0], cfg.steer_vector_norm, cfg.seed)  # once per run
        steer_vector, measure_vector = {  # (steer with, measure against)
            "persona": (persona_vector, persona_vector),
            "random": (random_vec, random_vec),
            "none": (None, persona_vector),
        }[cfg.steering_mode]
        results, histories = run_swarm(cfg, replicas, measure_vector, steer_vector)
        out_dir.mkdir(parents=True, exist_ok=True)
        config_record = {**dataclasses.asdict(cfg), "measured_vector": measured_vector_name(cfg)}
        (out_dir / f"{name}_config.json").write_text(json.dumps(config_record, indent=2))
        (out_dir / f"{name}_transcripts.json").write_text(json.dumps(histories, indent=2))
        results.to_csv(out_dir / f"{name}.csv", index=False)  # last: its existence marks the run done
        print(results.drop(columns=["system_prompt", "seed_prompt", "received", "text"]).to_string())


if __name__ == "__main__":
    main()
