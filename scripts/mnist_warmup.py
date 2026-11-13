"""Étape 0 : échauffement sur MNIST.

Entraîne un petit MLP (784, 256, 256 puis 10 neurones), puis mesure sa précision quand on
compresse les poids de ses trois couches nn.Linear avec quantize() et lowrank_approx().

Lancement, depuis la racine du dépôt :
    python -m scripts.mnist_warmup
"""
import copy
import time

import numpy as np
import torch
from datasets import load_dataset
from torch import nn

from src.lowrank import lowrank_approx
from src.quant import quantize

# 784 = 28 × 28 n'est pas divisible par 128 : on prend des groupes de 16,
# qui divisent à la fois 784 et 256.
GROUP = 16


def load_mnist(split):
    """Renvoie les images aplaties (float32 dans [0, 1]) et les étiquettes."""
    ds = load_dataset("ylecun/mnist", split=split)
    x = np.stack([np.asarray(img, dtype=np.uint8) for img in ds["image"]])
    x = torch.from_numpy(x).float().div(255).view(len(ds), -1)
    return x, torch.tensor(ds["label"])


def train(x, y, epochs=3, batch_size=128):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 10),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), batch_size):
            idx = perm[i : i + batch_size]
            loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def accuracy(model, x, y):
    return (model(x).argmax(dim=1) == y).float().mean().item()


@torch.no_grad()
def compress(model, fn):
    """Copie du modèle où le poids de chaque nn.Linear devient fn(poids).

    Les biais ne sont pas touchés : ils sont minuscules devant les poids.
    """
    model = copy.deepcopy(model)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            layer.weight.copy_(fn(layer.weight))
    return model


def main():
    x_train, y_train = load_mnist("train")
    x_test, y_test = load_mnist("test")

    start = time.time()
    model = train(x_train, y_train)
    print(f"Entraînement : {time.time() - start:.0f} s")
    print(f"Référence (fp32)         : {accuracy(model, x_test, y_test):.2%}")

    print("\nQuantification")
    for bits in [8, 4, 3, 2]:
        for gran in ["tensor", "row", GROUP]:
            compressed = compress(model, lambda w: quantize(w, bits, gran))
            print(f"  {bits} bits, {str(gran):>6}       : {accuracy(compressed, x_test, y_test):.2%}")

    print("\nRang faible")
    for ratio in [0.9, 0.75, 0.5, 0.25, 0.1]:
        compressed = compress(model, lambda w: lowrank_approx(w, ratio))
        print(f"  rank_ratio = {ratio:<4}      : {accuracy(compressed, x_test, y_test):.2%}")


if __name__ == "__main__":
    main()
