"""Grille de rang faible : rank_ratio ∈ {0,9 ; 0,75 ; 0,5 ; 0,25}.

Chaque expérience ajoute une ligne au CSV ; celles déjà présentes sont sautées.
La référence (modèle non compressé) est déjà dans results/quant.csv : on ne la
recalcule pas ici.

    python -m scripts.run_lowrank                                   # grille complète
    python -m scripts.run_lowrank --max-tokens 4096 --out results/quick_lowrank.csv
"""
import argparse
import time

from src.apply import compress_model, target_layers
from src.budget import lowrank_bits_per_param
from src.eval_ppl import load_model, load_wikitext_ids, perplexity
from src.lowrank import lowrank_approx
from src.results import append_result, is_done

RANK_RATIOS = [0.9, 0.75, 0.5, 0.25]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--out", default="results/lowrank.csv")
    args = parser.parse_args()

    model, tokenizer = load_model()
    ids = load_wikitext_ids(tokenizer, args.max_tokens)
    shapes = [tuple(layer.weight.shape) for _, layer in target_layers(model)]
    print(f"{len(shapes)} couches compressées, {len(ids)} tokens d'évaluation")

    for ratio in RANK_RATIOS:
        config = dict(method="lowrank", rank_ratio=ratio)
        if is_done(args.out, **config):
            print(f"rank_ratio = {ratio} : déjà fait")
            continue
        start = time.time()
        compressed = compress_model(model, lambda w: lowrank_approx(w, ratio))
        ppl = perplexity(compressed, ids)
        del compressed  # libère la mémoire du GPU avant la copie suivante
        bpp = lowrank_bits_per_param(shapes, ratio)
        append_result(args.out, **config, bits_per_param=round(bpp, 4),
                      ppl=ppl, seconds=round(time.time() - start, 1))
        print(f"rank_ratio = {ratio} : {bpp:.3f} bits/param, ppl = {ppl:.3f}")


if __name__ == "__main__":
    main()
