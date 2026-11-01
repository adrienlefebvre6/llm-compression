"""Grille de quantification : bits ∈ {8, 4, 3, 2} × granularité ∈ {tensor, row, 128}.

Chaque expérience ajoute une ligne au CSV ; celles déjà présentes sont sautées.
La première ligne est la référence (modèle non compressé).

    python -m scripts.run_quant                                   # grille complète
    python -m scripts.run_quant --max-tokens 4096 --out results/quick_quant.csv
"""
import argparse
import time

from src.apply import compress_model, target_layers
from src.budget import quant_bits_per_param
from src.eval_ppl import load_model, load_wikitext_ids, perplexity
from src.quant import quantize
from src.results import append_result, is_done

BITS = [8, 4, 3, 2]
GRANULARITIES = ["tensor", "row", 128]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--out", default="results/quant.csv")
    args = parser.parse_args()

    model, tokenizer = load_model()
    ids = load_wikitext_ids(tokenizer, args.max_tokens)
    shapes = [tuple(layer.weight.shape) for _, layer in target_layers(model)]
    print(f"{len(shapes)} couches compressées, {len(ids)} tokens d'évaluation")

    if not is_done(args.out, method="baseline", bits=16):
        start = time.time()
        ppl = perplexity(model, ids)
        append_result(args.out, method="baseline", bits=16, bits_per_param=16,
                      ppl=ppl, seconds=round(time.time() - start, 1))
        print(f"référence : ppl = {ppl:.3f}")

    for bits in BITS:
        for gran in GRANULARITIES:
            config = dict(method="quant", bits=bits, granularity=gran)
            if is_done(args.out, **config):
                print(f"{bits} bits, {gran} : déjà fait")
                continue
            start = time.time()
            compressed = compress_model(model, lambda w: quantize(w, bits, gran))
            ppl = perplexity(compressed, ids)
            del compressed  # libère la mémoire du GPU avant la copie suivante
            bpp = quant_bits_per_param(shapes, bits, gran)
            append_result(args.out, **config, bits_per_param=round(bpp, 4),
                          ppl=ppl, seconds=round(time.time() - start, 1))
            print(f"{bits} bits, {gran} : {bpp:.3f} bits/param, ppl = {ppl:.3f}")


if __name__ == "__main__":
    main()
