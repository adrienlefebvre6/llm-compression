"""Approximation de rang faible par SVD tronquée (écrite par Adrien)."""
import torch


def lowrank_approx(w: torch.Tensor, rank_ratio: float) -> torch.Tensor:
    """Remplace w par sa meilleure approximation de rang r (compression simulée).

    Args:
        w: matrice de poids (m × n).
        rank_ratio: fraction de paramètres conservée, en pratique entre 0 et 1.
            Le rang vaut r = max(1, ⌊rank_ratio · mn / (m + n)⌋), si bien que les
            deux facteurs U_r·diag(S_r) (m × r) et V_r^T (r × n) coûtent environ
            rank_ratio · mn paramètres. Une valeur > 1 est acceptée (les tests
            s'en servent pour atteindre le rang plein) : si r dépasse min(m, n),
            on garde simplement toutes les valeurs singulières.

    Méthode :
        SVD réduite en float32 : U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        puis on garde les r plus grandes valeurs singulières :
        ŵ = U_r · diag(S_r) · V_r^T

    Returns:
        ŵ, de même forme et de même type (dtype) que w.
    """
    raise NotImplementedError
