"""Lecture et écriture des CSV de résultats, une ligne par expérience."""
import csv
import os

COLUMNS = ["method", "bits", "granularity", "rank_ratio", "bits_per_param", "ppl", "seconds"]
KEY = ["method", "bits", "granularity", "rank_ratio"]


def _key(row):
    return tuple(str(row.get(c, "")) for c in KEY)


def done_keys(path):
    """Ensemble des expériences déjà présentes dans le CSV (vide s'il n'existe pas)."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {_key(row) for row in csv.DictReader(f)}


def is_done(path, **config):
    return _key(config) in done_keys(path)


def append_result(path, **row):
    """Ajoute une ligne tout de suite, pour ne rien perdre si Colab coupe."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in COLUMNS})
