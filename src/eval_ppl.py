"""Chargement de Qwen2.5-0.5B et perplexité sur WikiText-2 (partie test).

Perplexité de référence, depuis la racine du dépôt :
    python -m src.eval_ppl                     # texte complet (GPU conseillé)
    python -m src.eval_ppl --max-tokens 4096   # test rapide sur un extrait
"""
import argparse
import math
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def load_model():
    """Renvoie (modèle, tokenizer), en fp16 sur GPU.

    Sur CPU, on charge en float32 : les calculs en fp16 y sont très lents.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device)
    model.eval()
    return model, AutoTokenizer.from_pretrained(MODEL_NAME)


def load_wikitext_ids(tokenizer, max_tokens=None):
    """Textes de test joints par des sauts de ligne, tokenisés en une seule séquence."""
    ds = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1", split="test")
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    return ids[:max_tokens] if max_tokens else ids


@torch.no_grad()
def perplexity(model, ids, ctx_len=2048, stride=512):
    """Perplexité par fenêtre glissante.

    Chaque fenêtre fait ctx_len tokens et avance de stride tokens. Seuls les
    tokens nouveaux de chaque fenêtre (les stride derniers) comptent dans la
    perte : ils sont prédits avec au moins ctx_len - stride tokens de contexte.
    """
    device = next(model.parameters()).device
    nll_sum, n_tokens, prev_end = 0.0, 0, 0
    for begin in range(0, len(ids), stride):
        end = min(begin + ctx_len, len(ids))
        window = ids[begin:end].unsqueeze(0).to(device)
        logits = model(window).logits[0]
        # Le logit en position i prédit le token i + 1. Le tout premier token
        # du texte n'a pas de contexte, on ne le prédit pas.
        start = max(prev_end - begin, 1)
        nll_sum += torch.nn.functional.cross_entropy(
            logits[start - 1 : -1].float(), window[0, start:], reduction="sum"
        ).item()
        n_tokens += end - begin - start
        prev_end = end
        if end == len(ids):
            break
    return math.exp(nll_sum / n_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    model, tokenizer = load_model()
    ids = load_wikitext_ids(tokenizer, args.max_tokens)
    start = time.time()
    ppl = perplexity(model, ids)
    print(f"{len(ids)} tokens, perplexité = {ppl:.3f} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
