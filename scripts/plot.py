"""Perplexité en fonction des bits par paramètre, à partir des CSV de résultats.

    python -m scripts.plot                                # lit results/*.csv
    python -m scripts.plot --results-dir /content/drive/MyDrive/llm-compression/results
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

# Une couleur et un marqueur par série, toujours dans le même ordre : la série
# reste reconnaissable même sans les couleurs (impression, daltonisme).
SERIES = [
    ("quant", "tensor", "Quantification, 1 échelle / matrice", "#2a78d6", "o"),
    ("quant", "row", "Quantification, 1 échelle / ligne", "#eb6834", "s"),
    ("quant", "128", "Quantification, groupes de 128", "#1baf7a", "^"),
    ("lowrank", None, "Rang faible (SVD tronquée)", "#eda100", "D"),
]
UNUSABLE_PPL = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="figures/ppl_vs_bits.png")
    args = parser.parse_args()

    files = [f for f in glob.glob(os.path.join(args.results_dir, "*.csv"))
             if not os.path.basename(f).startswith("quick_")]
    df = pd.concat([pd.read_csv(f, dtype={"granularity": str}) for f in files])

    fig, ax = plt.subplots(figsize=(8, 5.8))
    for method, gran, label, color, marker in SERIES:
        rows = df[df.method == method]
        if gran is not None:
            rows = rows[rows.granularity == gran]
        if rows.empty:
            continue
        rows = rows.sort_values("bits_per_param")
        ax.plot(rows.bits_per_param, rows.ppl, color=color, marker=marker,
                markersize=8, linewidth=2, label=label)

    baseline = df[df.method == "baseline"]
    if not baseline.empty:
        ppl = baseline.ppl.iloc[0]
        ax.axhline(ppl, color="#52514e", linestyle="--", linewidth=1.5,
                   label=f"Modèle d'origine, fp16 (ppl = {ppl:.2f})")

    # Au-delà de ~1000, le modèle ne produit plus que du bruit : les écarts
    # entre configurations n'y ont plus de sens.
    ax.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    ax.axhspan(UNUSABLE_PPL, ymax, color="#f0efec", zorder=0)
    ax.set_ylim(ymin, ymax)
    ax.text(ax.get_xlim()[1], UNUSABLE_PPL * 1.3, "modèle inutilisable ",
            color="#52514e", fontsize=9, ha="right", va="bottom")

    ax.set_xlabel("Bits par paramètre (couches compressées)")
    ax.set_ylabel("Perplexité WikiText-2 (échelle log)")
    ax.set_title("Qwen2.5-0.5B : qualité en fonction de la taille")
    ax.grid(True, which="major", color="#e5e5e2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Figure enregistrée : {args.out}")


if __name__ == "__main__":
    main()
