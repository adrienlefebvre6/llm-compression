"""Étape 3 : ce que les poids de Qwen ont de particulier.

Deux figures, sans faire tourner le modèle (seuls les poids sont lus) :
- figures/spectrum.png : valeurs singulières de q_proj, v_proj et down_proj au
  début, au milieu et à la fin du modèle, comparées à une matrice aléatoire ;
- figures/outliers.png : rapport max|w| / écart-type de chaque couche compressée.

Tourne sur CPU en une minute environ :
    python -m scripts.analyze_weights
"""
import os

import matplotlib.pyplot as plt
import torch

from src.apply import TARGETS, target_layers
from src.eval_ppl import load_model

BLOCKS = [0, 12, 23]
SPECTRUM_TARGETS = ["q_proj", "v_proj", "down_proj"]
# Profondeur ordonnée : une seule teinte, du clair (début) au foncé (fin).
DEPTH_COLORS = ["#86b6ef", "#2a78d6", "#104281"]
TARGET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MUTED = "#52514e"


def peak_ratio(w):
    """max|w| / écart-type : environ 5 pour une gaussienne de cette taille."""
    return (w.abs().max() / w.std()).item()


def kept_energy(s, rank_ratio, m, n):
    """Part de Σσ² conservée par lowrank_approx : 1 - (erreur relative)²."""
    r = max(1, int(rank_ratio * m * n / (m + n)))
    return (s[:r] ** 2).sum().item() / (s**2).sum().item()


def plot_spectrum(layers, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    print("\nÉnergie conservée par lowrank_approx (rank_ratio = 0,5 / 0,25)")
    for ax, kind in zip(axes, SPECTRUM_TARGETS):
        for block, color in zip(BLOCKS, DEPTH_COLORS):
            w = layers[(block, kind)].float()
            s = torch.linalg.svdvals(w)
            x = torch.arange(len(s)) / len(s)
            ax.plot(x, s / s[0], color=color, linewidth=2, label=f"bloc {block}")
            m, n = w.shape
            print(f"  {kind:>9}, bloc {block:>2} : {kept_energy(s, 0.5, m, n):.0%} / "
                  f"{kept_energy(s, 0.25, m, n):.0%}")

        # Référence : matrice gaussienne de même forme, dont le spectre est plat.
        m, n = w.shape
        s_rand = torch.linalg.svdvals(torch.randn(m, n, generator=torch.Generator().manual_seed(0)))
        ax.plot(torch.arange(len(s_rand)) / len(s_rand), s_rand / s_rand[0],
                color=MUTED, linestyle="--", linewidth=1.5, label="matrice aléatoire")
        print(f"  {kind:>9}, aléatoire : {kept_energy(s_rand, 0.5, m, n):.0%} / "
              f"{kept_energy(s_rand, 0.25, m, n):.0%}")

        # Où coupe rank_ratio = 0,5 : tout ce qui est à droite est jeté.
        cut = int(0.5 * m * n / (m + n)) / min(m, n)
        ax.axvline(cut, color=MUTED, linestyle=":", linewidth=1)
        ax.text(cut, 2e-6, " coupure\n rank_ratio = 0,5", color=MUTED, fontsize=8)

        ax.set_title(f"{kind} ({m} × {n})")
        ax.set_xlabel("Position i de la valeur singulière σᵢ\n(fraction de min(m, n))", fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, which="major", color="#e5e5e2", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("σᵢ / σ₁ (échelle log)")
    axes[2].legend(frameon=False, loc="center right")
    fig.suptitle("Qwen2.5-0.5B : spectre des valeurs singulières")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Figure enregistrée : {out}")


def plot_outliers(layers, n_blocks, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    ratios = {key: peak_ratio(w.float()) for key, w in layers.items()}
    for kind, color in zip(TARGETS, TARGET_COLORS):
        ax.plot(range(n_blocks), [ratios[(b, kind)] for b in range(n_blocks)],
                color=color, marker="o", markersize=4, linewidth=1.5, label=kind)

    gauss = peak_ratio(torch.randn(896, 896, generator=torch.Generator().manual_seed(0)))
    ax.axhline(gauss, color=MUTED, linestyle="--", linewidth=1.5,
               label=f"matrice gaussienne ({gauss:.1f})")

    ax.set_yscale("log")
    ax.set_xlabel("Bloc")
    ax.set_ylabel("max|w| / écart-type (échelle log)")
    ax.set_title("Qwen2.5-0.5B : poids extrêmes par couche")
    ax.grid(True, which="major", color="#e5e5e2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Figure enregistrée : {out}")

    print("\nCouches aux poids les plus extrêmes (max|w| / écart-type)")
    for (block, kind), ratio in sorted(ratios.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  bloc {block:>2}, {kind:>9} : {ratio:.0f}")


def main():
    model, _ = load_model()
    # Les noms ressemblent à "model.layers.12.mlp.down_proj" : on garde (12, "down_proj").
    layers = {
        (int(name.split(".")[2]), name.split(".")[-1]): layer.weight.detach()
        for name, layer in target_layers(model)
    }
    os.makedirs("figures", exist_ok=True)
    plot_spectrum(layers, "figures/spectrum.png")
    plot_outliers(layers, model.config.num_hidden_layers, "figures/outliers.png")


if __name__ == "__main__":
    main()
