# country2vec

Color a world or US map by how semantically close a word is to each country (or
state) name. Built on Google News Word2Vec — the names of countries (and US
states) end up acting as a fuzzy choropleth of meaning. "vodka" lights up
Eastern Europe, "malaria" the African belt, "cowboy" the western US.

Live at [douwe.com/projects/mapof](https://douwe.com/projects/mapof). This repo
is also the rendered project directory inside the larger
[douweosinga/djangosite](https://github.com/DOsinga/douweosinga) site — the same
files run both ways.

## Run

```
pip install -r requirements.txt
python build_db.py
```

`build_db.py` fetches the 3M-word Google News word2vec model via gensim (~1.5 GB
download, cached under `~/gensim-data/`), collapses every word's case-variants
into a single float32 vector per lowercase word, and writes them to
`static/word2vec.db` (~3.3 GB) as a sqlite-vec `vec0` virtual table. A small
`words_lower` index is also written for autocomplete prefix lookups. About four
minutes of work after the download finishes.

The Django view (`mapof.py`) reads the DB through `apsw + sqlite-vec` and renders
a Plotly choropleth — `natural earth` projection for the world map,
`albers usa` for the US map. Map type is selected via `?map=world` (default) or
`?map=usa`; adding a third map type means appending an entry to the
`COUNTRIES`/`STATES`-style dicts in `mapof.py`.

## How it works

For each lowercase word, the stored vector is the sum of the word's lowercase
and Titlecase variants in Google News. The ALL-CAPS variant is treated as noise
and skipped unless it's the only form present. Querying then becomes a single
primary-key lookup per word — both the user's word and each country (or US
state) name resolve to one vector apiece, and cosine distance between them is
used as the choropleth value (rescaled per map to [0, 1]).

The original (2016) version of this project imported the same data into Postgres
and queried with the `cube` extension. The current code reads from sqlite-vec,
so it has no Postgres dependency and ships the DB as a single file.

## Files

- `mapof.py` — Django view; the `MapOf.fill_dict` method drives both maps.
- `mapof.html` — Plotly choropleth template; transparent background.
- `build_db.py` — produces `static/word2vec.db` from Google News word2vec.
- `requirements.txt` — minimal deps for the standalone build / run.
- `static/word2vec.db` — generated, not tracked in git.
