"""Tests de quantize() : ils passent quand src/quant.py est correct."""
import pytest
import torch

from src.quant import quantize

GRANULARITIES = ["tensor", "row", 128]


def rel_error(w, w_hat):
    """Erreur relative ||w - ŵ|| / ||w||, calculée en float32."""
    w, w_hat = w.float(), w_hat.float()
    return ((w - w_hat).norm() / w.norm()).item()


def random_weights(m=256, n=512, dtype=torch.float16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, n, generator=g).to(dtype)


def strictly_decreasing(values):
    return all(a > b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_shape_and_dtype(granularity, dtype):
    w = random_weights(dtype=dtype)
    w_hat = quantize(w, 4, granularity)
    assert w_hat.shape == w.shape
    assert w_hat.dtype == w.dtype


def test_input_is_not_modified():
    w = random_weights()
    w_copy = w.clone()
    quantize(w, 2, 128)
    assert torch.equal(w, w_copy)


@pytest.mark.parametrize("bits", [8, 4, 3, 2])
@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_values_on_grid_are_unchanged(granularity, bits):
    # w = s · q, avec q entier dans [-q_max, q_max] et s une puissance de 2 :
    # tout est exact en fp16. On place q_max au début de chaque groupe de 128
    # pour que max|w_groupe| / q_max retombe exactement sur s partout.
    q_max = 2 ** (bits - 1) - 1
    g = torch.Generator().manual_seed(0)
    q = torch.randint(-q_max, q_max + 1, (64, 256), generator=g)
    q[:, ::128] = q_max
    w = (q * 2.0**-6).to(torch.float16)
    assert torch.equal(quantize(w, bits, granularity), w)


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_error_decreases_with_bits(granularity):
    w = random_weights()
    errors = [rel_error(w, quantize(w, b, granularity)) for b in [2, 3, 4, 8]]
    assert strictly_decreasing(errors), errors


@pytest.mark.parametrize("bits", [8, 4, 3, 2])
def test_error_decreases_with_smaller_groups(bits):
    # Plus le groupe est petit, plus son max est proche des valeurs typiques,
    # donc plus la grille est fine.
    w = random_weights()
    errors = [rel_error(w, quantize(w, bits, gr)) for gr in ["tensor", "row", 128, 32]]
    assert strictly_decreasing(errors), errors


@pytest.mark.parametrize("granularity", ["row", 128])
def test_8_bits_relative_error_below_1_percent(granularity):
    # "tensor" est exclu volontairement : sur une gaussienne de 131 000 valeurs,
    # le max vaut environ 4,4 écarts-types, et l'erreur tombe pile autour de 1 %.
    # C'est le problème des valeurs extrêmes, que les petites échelles corrigent.
    w = random_weights()
    assert rel_error(w, quantize(w, 8, granularity)) < 0.01


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_zero_groups_do_not_produce_nan(granularity):
    w = random_weights(64, 256)
    w[0] = 0  # une ligne entièrement nulle
    w_hat = quantize(w, 4, granularity)
    assert not torch.isnan(w_hat).any()
    assert torch.equal(w_hat[0], w[0])

    zeros = torch.zeros_like(w)
    assert torch.equal(quantize(zeros, 4, granularity), zeros)
