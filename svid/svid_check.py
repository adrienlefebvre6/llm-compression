"""Vérification de SVID (Tran & Nguyen, ICLR 2026, arXiv 2505.22811) sur Qwen2.5-0.5B.

SVID approxime W par sign(W) ⊙ (rang 1 de |W|) : on garde le signe de chaque
poids et on approxime leurs tailles par un rang 1. Avec plusieurs noyaux, chaque
nouveau noyau approxime le résidu laissé par les précédents.

Pour chaque matrice compressée de Qwen, et pour une matrice gaussienne de même
forme, on mesure l'erreur relative ||W - Ŵ|| / ||W|| de :
- un rang 1 direct de W ;
- SVID avec 1, 2 et 3 noyaux (environ 1 bit par poids et par noyau) ;
- une SVD tronquée de W qui utilise autant de bits qu'un noyau SVID.

Tourne sur CPU en quelques minutes (seuls les poids sont lus), depuis la racine du dépôt :
    python -m svid.svid_check
"""
import pandas as pd
import torch

from src.apply import target_layers
from src.eval_ppl import load_model

OUT = "svid/results.csv"


def rel_error(w, w_hat):
    return ((w - w_hat).norm() / w.norm()).item()


def rank1(w):
    """Meilleure approximation de rang 1 de w, par SVD."""
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    return s[0] * torch.outer(u[:, 0], vh[0])


def svid(w):
    """Un noyau SVID : les signes de w, multipliés par le rang 1 de ses tailles |w|."""
    return torch.sign(w) * rank1(w.abs())


def measure(w):
    """Mesure, pour une matrice w, l'erreur relative de chaque approximation.

    Renvoie un dictionnaire : une entrée pour le rang 1 direct, une pour SVID
    avec 1, 2 et 3 noyaux, et une pour la SVD tronquée au même nombre de bits.
    """
    m, n = w.shape
    row = dict(m=m, n=n, err_rank1=rel_error(w, rank1(w)))

    approx = torch.zeros_like(w)
    for k in [1, 2, 3]:
        approx = approx + svid(w - approx)  # le noyau k approxime le résidu
        row[f"err_svid{k}"] = rel_error(w, approx)

    # Un noyau coûte mn bits de signe + (m + n) échelles fp16 ; un rang r coûte
    # 16·r·(m + n) bits. On garde le plus grand r qui tient dans le même budget.
    r = (m * n + 16 * (m + n)) // (16 * (m + n))
    s = torch.linalg.svdvals(w)
    row["err_svd_same_bits"] = ((s[r:] ** 2).sum() / (s**2).sum()).sqrt().item()
    return row


def main():
    """Mesure les 168 matrices compressées de Qwen, plus une gaussienne par type de
    matrice, puis enregistre le CSV et affiche les moyennes par type."""
    model, _ = load_model()
    g = torch.Generator().manual_seed(0)
    rows = []
    for name, layer in target_layers(model):
        # Dans "model.layers.12.mlp.down_proj", le bloc est 12 et la matrice down_proj.
        block, matrix = name.split(".")[2], name.split(".")[-1]
        w = layer.weight.detach().float()
        rows.append(dict(block=block, matrix=matrix, **measure(w)))
        if block == "0":  # référence : une gaussienne de même forme par type de matrice
            rows.append(dict(block="random", matrix=matrix,
                             **measure(torch.randn(w.shape, generator=g))))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    df["source"] = df.block.where(df.block == "random", "qwen")
    print(df.groupby(["matrix", "source"])[["err_rank1", "err_svid1", "err_svid2", "err_svid3",
                                            "err_svd_same_bits"]].mean().round(3))
    print(f"Résultats enregistrés : {OUT}")


if __name__ == "__main__":
    main()
