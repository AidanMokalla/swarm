"""Interactive: python plot_results.py -> prompts for metric, x axis, filters, color and facet.

Varying metadata columns you don't filter, color by, or facet on are averaged over.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
import pandas as pd
import seaborn as sns

from config import CFG

METADATA = ["steering_mode", "persona_trait", "layer", "alpha", "steer_vector_norm", "n_agents", "topology", "n_turns",
            "exchanges_per_edge", "temperature", "seed", "hop_isolated", "source_agents"]
# all metrics use the UNIT measured vector (cos / projection), so they compare across modes;
# in 'random' mode the measured vector is the random one (see the `measured_vector` column)
METRICS = ["prompt_proj", "prompt_cos", "gen_proj", "gen_cos"]
X_AXES = ["hop", "turn", "exchange"]  # rows of all exchanges/partners are averaged unless x = exchange


def ask(question, default=""):
    answer = input(f"{question} [{default}]: " if default else f"{question}: ").strip()
    return answer or default


def ask_choice(question, options, default):
    """Pick one of options by name or number; `default` may be None (-> 'none')."""
    shown = ", ".join(f"{i}={o}" for i, o in enumerate(options))
    while True:
        answer = ask(f"{question} ({shown})", default or "none")
        if answer == "none" and default is None:
            return None
        if answer in options:
            return answer
        if answer.isdigit() and int(answer) < len(options):
            return options[int(answer)]
        print("  invalid choice")


def varying_columns(df):
    return [c for c in METADATA if c in df and df[c].nunique() > 1]


def filter_values(df):
    """For each varying column, optionally keep a subset of values (enter = all)."""
    for column in varying_columns(df):
        values = sorted(df[column].astype(str).unique())
        answer = ask(f"keep which {column}? (comma-separated from {values}; enter = all)")
        if answer:
            keep = [v.strip() for v in answer.split(",")]
            df = df[df[column].astype(str).isin(keep)]
            if df.empty:
                raise SystemExit(f"no rows left after filtering {column} to {keep}")
    return df


def make_plot(results, out_dir):
    y = ask_choice("metric", METRICS, "gen_cos")
    x = ask_choice("x axis", X_AXES, "hop")
    if ask("include source agent (hop 0)? y/n", "n").lower() != "y":
        results = results.query("hop > 0")
    results = filter_values(results)
    varying = varying_columns(results)
    hue = ask_choice("color by", varying, "steering_mode" if "steering_mode" in varying else None) if varying else None
    left = [c for c in varying if c != hue]
    col = ask_choice("facet columns (one panel per value) by", left, "steering_mode" if "steering_mode" in left else None) if left else None
    left = [c for c in left if c != col]
    row = ask_choice("facet rows by", left, None) if left else None
    averaged = [c for c in left if c != row]
    print(f"averaging over: {averaged or 'nothing'}")

    results = results.assign(**{c: results[c].astype(str) for c in (hue, col, row) if c})
    sns.set_theme(style="whitegrid", palette="colorblind")
    grid = sns.relplot(data=results, x=x, y=y, hue=hue, col=col, row=row,
                       kind="line", marker="o", height=4, aspect=1.1)
    grid.set_axis_labels(x, y)
    path = out_dir / ask("filename", f"{y}_vs_{x}_hue-{hue}_col-{col}_row-{row}.png")
    grid.savefig(path, dpi=150)
    print(f"saved {path}")
    if ask("show plot? y/n", "n").lower() == "y":
        plt.show()
    plt.close("all")


def plot_network(results, metric, path, by_turn=False, baseline=None):
    """undirected networkx graph per run
    width ~ messages exchanged, dark in turns where the pair talked.
    Source agents get a black outline. The color range is set by non-source agents only
    by_turn: break it down per turn.
    baseline: rows of a reference run (here alpha=0); subtract its mean"""
    label, heading = f"mean {metric}", ""
    if baseline is not None:
        ref = baseline.groupby(["turn", "agent"])[metric].mean().rename("ref")
        results = results.join(ref, on=["turn", "agent"]).assign(**{metric: lambda d: d[metric] - d.ref})
        differs = [c for c in METADATA if c in baseline and baseline[c].iloc[0] != results[c].iloc[0]]
        heading = f"relative to baseline ({', '.join(f'{c}={baseline[c].iloc[0]}' for c in differs)})"
        label = f"{metric} relative to baseline"
    runs = varying_columns(results)
    groups = list(results.groupby(runs)) if runs else [((), results)]
    lim = results[results.hop > 0].groupby(runs + ["turn", "agent"])[metric].mean().abs().max()
    norm = Normalize(-lim, lim)
    cmap = plt.get_cmap("coolwarm")
    turns = sorted(results.turn.unique()) if by_turn else [None]
    width = 10 if len(groups) == 1 else 6
    fig, axes = plt.subplots(len(turns), len(groups), squeeze=False,
                             figsize=(width * len(groups), (1.6 if by_turn else 3.2) * len(turns) + 0.8))
    fig.subplots_adjust(hspace=0.6, wspace=0.05)
    for j, (key, run) in enumerate(groups):
        hops = run.groupby("agent").hop.first()
        steered = run.groupby("agent").steered.any()
        talk = run[run.partner >= 0]
        edges = talk.groupby([talk[["partner", "agent"]].min(axis=1), talk[["partner", "agent"]].max(axis=1)]).size()
        graph = nx.Graph()
        graph.add_nodes_from((agent, {"hop": hop}) for agent, hop in hops.items())
        graph.add_weighted_edges_from((src, dst, count) for (src, dst), count in edges.items())
        pos = nx.multipartite_layout(graph, subset_key="hop")
        key = key if isinstance(key, tuple) else (key,)
        title = ", ".join(f"{c}={v}" for c, v in zip(runs, key)) or ", ".join(
            f"{c}={run[c].iloc[0]}" for c in ["steering_mode", "persona_trait", "alpha", "topology", "exchanges_per_edge"])
        if len(groups) == 1:
            heading = f"{title}\n{heading}".strip()
        for i, turn in enumerate(turns):
            ax = axes[i, j]
            if turn is None:
                now, latest = run, run.groupby("agent")[metric].mean()
            else:
                upto, now = run[run.turn <= turn], run[run.turn == turn]
                latest = upto[upto.turn == upto.groupby("agent").turn.transform("max")].groupby("agent")[metric].mean()
            active = {frozenset(e) for e in zip(now.partner, now.agent)}
            nx.draw_networkx_edges(graph, pos, ax=ax, node_size=500,
                                   edge_color=["dimgray" if frozenset(e) in active else "lightgray" for e in graph.edges],
                                   width=[0.5 + w / edges.max() * 2 for *_, w in graph.edges(data="weight")])
            nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=500,
                                   node_color=[cmap(norm(latest[a])) if a in latest else "white" for a in graph],
                                   linewidths=2,
                                   edgecolors=["black" if steered[a] else "lightgray" if a not in latest else "none"
                                               for a in graph])
            nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
            sub = "" if turn is None else f"turn {turn}"
            if len(groups) > 1 and i == 0:  # name each run once, above its column
                sub = f"{title}\n{sub}".strip()
            ax.set_title(sub, fontsize=9)
            ax.axis("off")
    fig.suptitle(heading, fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label=label, shrink=0.6 if by_turn else 1.0, extend="both")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    plt.close(fig)


def main():
    out_dir = Path(CFG.output_dir)
    paths = sorted(out_dir.glob("*.csv"))
    if not paths:
        raise SystemExit(f"no csv files in {out_dir}")
    results = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    print(f"loaded {len(paths)} runs, {len(results)} rows")
    while True:
        if ask_choice("plot type", ["line", "network"], "line") == "network":
            metric = ask_choice("metric", METRICS, "gen_cos")
            by_turn = ask("one row per turn? y/n", "n").lower() == "y"
            name = f"network_{metric}{'_by_turn' if by_turn else ''}.png"
            plot_network(filter_values(results), metric, out_dir / ask("filename", name), by_turn)
        else:
            make_plot(results, out_dir)
        if ask("another plot? y/n", "n").lower() != "y":
            break


if __name__ == "__main__":
    main()
