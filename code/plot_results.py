"""Optional: python plot_results.py -> projection vs hop and vs turn, per steering mode."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CFG


def main():
    out_dir = Path(CFG.output_dir)
    results = pd.concat(pd.read_csv(p) for p in out_dir.glob("*.csv"))
    others = results.query("hop > 0")
    sns.set_theme(style="whitegrid", palette="colorblind")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, x in zip(axes, ["hop", "turn"]):
        sns.lineplot(data=others, x=x, y="prompt_proj", hue="steering_mode", marker="o", ax=ax)
        ax.set(xlabel=x, ylabel="last-prompt-token projection on v̂")
    fig.tight_layout()
    fig.savefig(out_dir / "projection.png", dpi=150)


if __name__ == "__main__":
    main()
