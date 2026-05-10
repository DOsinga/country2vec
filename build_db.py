"""Build static/word2vec.db from Google News word2vec.

For each lowercase word, sums the vectors of its case-variants
(lowercase + Titlecase, falling back to ALL-CAPS only when neither
of those exists). The result is one float32 vector per lowercase
word, written to a sqlite-vec ``vec0`` virtual table. A small
``words_lower`` index is also written for autocomplete prefix lookups.

Run once after cloning:

    pip install -r requirements.txt
    python build_db.py

This fetches the 3M-word Google News pretrained model via gensim
(~1.5 GB download, cached under ``~/gensim-data/``) and produces a
~3.3 GB output. About four minutes of work after the download.
"""
import argparse
import os
import time

import apsw
import gensim.downloader as gd_api
import numpy as np
import sqlite_vec
from gensim.models import KeyedVectors

DIM = 300
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "static", "word2vec.db")


def load_model(source: str) -> KeyedVectors:
    if os.path.exists(source):
        print(f"loading {source}...")
        return KeyedVectors.load_word2vec_format(source, binary=True)
    print(f"loading {source!r} via gensim.downloader (cached under ~/gensim-data)...")
    return gd_api.load(source)


def aggregate(kv: KeyedVectors) -> dict[str, np.ndarray]:
    """Returns {lower_word: summed_vec}."""
    sums: dict[str, list] = {}
    t0 = time.time()
    n = len(kv.index_to_key)
    for i, word in enumerate(kv.index_to_key):
        if i and i % 500_000 == 0:
            print(f"  {i:,}/{n:,} ({time.time()-t0:.1f}s)", flush=True)
        lower = word.lower()
        v = np.asarray(kv.vectors[i], dtype=np.float32)
        is_primary = not (word == word.upper() and word != lower)
        entry = sums.get(lower)
        if entry is None:
            sums[lower] = [v.copy(), is_primary]
        else:
            cur_v, cur_primary = entry
            if is_primary and cur_primary:
                entry[0] = cur_v + v
            elif is_primary and not cur_primary:
                entry[0] = v.copy()
                entry[1] = True
            # else: keep existing (primary already set, or only ALL-CAPS so far)
    print(f"  aggregated to {len(sums):,} lowercase entries ({time.time()-t0:.1f}s)")
    return {k: v[0] for k, v in sums.items()}


def write_db(vectors: dict[str, np.ndarray], out_path: str) -> None:
    if os.path.exists(out_path):
        os.remove(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    conn = apsw.Connection(out_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute(
        f"CREATE VIRTUAL TABLE embeddings USING vec0(word TEXT PRIMARY KEY, embedding float[{DIM}])"
    )
    conn.execute("CREATE TABLE words_lower (word_lower TEXT PRIMARY KEY)")

    t0 = time.time()
    conn.execute("BEGIN")
    conn.executemany(
        "INSERT INTO embeddings(word, embedding) VALUES (?, ?)",
        ((k, v.tobytes()) for k, v in vectors.items()),
    )
    conn.executemany(
        "INSERT INTO words_lower(word_lower) VALUES (?)", ((k,) for k in vectors)
    )
    conn.execute("COMMIT")
    print(f"  wrote {len(vectors):,} rows in {time.time()-t0:.1f}s; running VACUUM")
    conn.execute("VACUUM")
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  {out_path}: {size_mb:,.0f} MB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=DEFAULT_OUT, help="output sqlite DB path")
    p.add_argument(
        "--source",
        default="word2vec-google-news-300",
        help="gensim model name or path to a word2vec .bin file",
    )
    args = p.parse_args()

    kv = load_model(args.source)
    if kv.vector_size != DIM:
        raise SystemExit(f"expected {DIM}-dim vectors, got {kv.vector_size}")
    print(f"  {len(kv.index_to_key):,} words")

    vectors = aggregate(kv)
    write_db(vectors, args.out)


if __name__ == "__main__":
    main()
