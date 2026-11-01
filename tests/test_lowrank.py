"""Tests de lowrank_approx() : ils passent quand src/lowrank.py est correct."""
import pytest
import torch

from src.lowrank import lowrank_approx


def rel_error(w, w_hat):
    """Erreur relative ||w - ŵ|| / ||w||, calculée en float32."""
    w, w_hat = w.float(), w_hat.float()
    return ((w - w_hat).norm() / w.norm()).item()


def random_weights(m, n, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, n, generator=g).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(64, 96), (96, 64)])
def test_shape_and_dtype(shape, dtype):
    w = random_weights(*shape, dtype=dtype)
    w_hat = lowrank_approx(w, 0.5)
    assert w_hat.shape == w.shape
    assert w_hat.dtype == w.dtype


def test_output_has_expected_rank():
    # m = 128, n = 192 : r = ⌊0,5 × 24576 / 320⌋ = ⌊38,4⌋ = 38
    w = random_weights(128, 192)
    assert torch.linalg.matrix_rank(lowrank_approx(w, 0.5)).item() == 38


def test_rank_is_at_least_one():
    # ⌊1e-6 × mn / (m + n)⌋ = 0, mais on garde au moins la plus grande valeur singulière.
    w = random_weights(64, 96)
    assert torch.linalg.matrix_rank(lowrank_approx(w, 1e-6)).item() == 1


@pytest.mark.parametrize("shape", [(64, 64), (64, 96), (96, 64)])
def test_full_rank_reconstructs_matrix(shape):
    # Avec rank_ratio ≤ 1, on n'atteint jamais le rang plein : pour r = min(m, n),
    # les facteurs coûtent r(m + n) > mn paramètres. On passe donc rank_ratio = 3,
    # qui donne r ≥ min(m, n), c'est-à-dire toutes les valeurs singulières.
    w = random_weights(*shape)
    assert torch.allclose(lowrank_approx(w, 3.0), w, atol=1e-4)


def test_error_decreases_with_rank_ratio():
    w = random_weights(128, 192)
    errors = [rel_error(w, lowrank_approx(w, r)) for r in [0.1, 0.25, 0.5, 0.75, 0.9]]
    assert all(a > b for a, b in zip(errors, errors[1:])), errors


def test_rank_5_matrix_is_reconstructed_exactly():
    # w = A·B, de rang 5. Ici mn / (m + n) = 8000 / 180 ≈ 44,4 :
    # rank_ratio = 0,12 donne r = 5, et 0,1 donne r = 4.
    w = random_weights(100, 5, seed=0) @ random_weights(5, 80, seed=1)
    for ratio in [0.12, 0.5, 0.9]:
        assert torch.allclose(lowrank_approx(w, ratio), w, atol=1e-4), ratio
    # Avec r = 4, il manque une direction : la reconstruction n'est plus exacte.
    assert rel_error(w, lowrank_approx(w, 0.1)) > 1e-2
