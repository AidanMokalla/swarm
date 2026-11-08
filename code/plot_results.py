"""Interactive: python plot_results.py -> prompts for metric, x axis, filters, color and facet.

Varying metadata columns you don't filter, color by, or facet on are averaged over.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CFG

METADATA = ["steering_mode", "persona_trait", "layer", "alpha", "n_agents", "topology", "n_turns",
            "temperature", "seed", "hop_isolated", "source_agents"]
METRICS = ["prompt_proj", "prompt_cos", "gen_proj", "gen_cos"]
X_AXES = ["hop", "turn"]


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
    y = ask_choice("metric", METRICS, "prompt_proj")
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


def main():
    out_dir = Path(CFG.output_dir)
    paths = sorted(out_dir.glob("*.csv"))
    if not paths:
        raise SystemExit(f"no csv files in {out_dir}")
    results = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    print(f"loaded {len(paths)} runs, {len(results)} rows")
    while True:
        make_plot(results, out_dir)
        if ask("another plot? y/n", "n").lower() != "y":
            break


if __name__ == "__main__":
    main()
