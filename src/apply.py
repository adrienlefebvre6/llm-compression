"""Applique une fonction de compression à toutes les couches ciblées du modèle."""
import copy

import torch

# Les 7 couches nn.Linear de chaque bloc. Les embeddings et la tête de sortie
# (partagées dans Qwen2.5-0.5B) ne sont pas touchés.
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def target_layers(model):
    """Liste des (nom, couche) compressées, dans l'ordre du modèle."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in TARGETS
    ]


@torch.no_grad()
def compress_model(model, fn):
    """Copie du modèle où le poids w de chaque couche ciblée devient fn(w).

    Le modèle d'origine reste intact, ce qui permet d'enchaîner les expériences
    sans le recharger. Les biais (présents sur q, k, v) ne sont pas compressés.
    """
    model = copy.deepcopy(model)
    for _, layer in target_layers(model):
        layer.weight.copy_(fn(layer.weight))
    return model
