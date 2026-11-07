"""Optional: python plot_results.py -> projection vs hop / vs turn (hue=steering_mode) for all runs.

Facets on CFG.plot_facet; other varying metadata columns are averaged over (printed below).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CFG

METADATA = ["persona_trait", "layer", "alpha", "n_agents", "topology", "n_turns",
            "temperature", "seed", "hop_isolated", "source_agents"]


def main():
    out_dir = Path(CFG.output_dir)
    results = pd.concat((pd.read_csv(p) for p in sorted(out_dir.glob("*.csv"))), ignore_index=True)
    others = results.query("hop > 0")
    varying = [c for c in METADATA if c in others and others[c].nunique() > 1]
    print("varying:", varying, "| averaged over:", [c for c in varying if c != CFG.plot_facet])
    sns.set_theme(style="whitegrid", palette="colorblind")
    for x in ["hop", "turn"]:
        grid = sns.relplot(data=others, x=x, y="prompt_proj", hue="steering_mode", col=CFG.plot_facet,
                           kind="line", marker="o", height=4, aspect=1.1)
        grid.set_axis_labels(x, "last-prompt-token projection on v̂")
        grid.savefig(out_dir / f"projection_{x}.png", dpi=150)


if __name__ == "__main__":
    main()
