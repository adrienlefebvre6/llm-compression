"""Bits par paramètre de chaque configuration, sur les couches compressées.

On compte exactement ce qu'il faudrait stocker pour chaque matrice (m × n),
puis on divise par le nombre total de poids d'origine. Les embeddings, non
compressés, ne sont pas comptés.
"""


def quant_bits_per_param(shapes, bits, granularity):
    """b bits par poids, plus une échelle fp16 (16 bits) par groupe.

    Revient à b + 16/g pour des groupes de g, b + 16/n par ligne, ≈ b pour la matrice.
    """
    total_bits = total_params = 0
    for m, n in shapes:
        if granularity == "tensor":
            n_scales = 1
        elif granularity == "row":
            n_scales = m
        else:
            n_scales = m * n // granularity
        total_bits += bits * m * n + 16 * n_scales
        total_params += m * n
    return total_bits / total_params


def lowrank_bits_per_param(shapes, rank_ratio):
    """Deux facteurs fp16 de r(m + n) poids au total, au lieu de m × n.

    Revient à ≈ 16 × rank_ratio (un peu moins, à cause de l'arrondi de r).
    """
    total_bits = total_params = 0
    for m, n in shapes:
        r = max(1, int(rank_ratio * m * n / (m + n)))
        total_bits += 16 * r * (m + n)
        total_params += m * n
    return total_bits / total_params
