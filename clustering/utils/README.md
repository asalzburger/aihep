# clustering_utils

Shared CSV/Apache Arrow table read/write for the `clustering/` packages
([`sensor`](../sensor), [`tracker`](../tracker)). Both had a byte-for-byte
identical `write_table`/`read_table` (and near-identical `write_run`/
`read_run`, differing only in which named tables they bundle) -- this
package is that logic, factored out once.

## API

- `write_table(df, path, fmt)` / `read_table(path, fmt)` -- one table, `fmt`
  is `"csv"` or `"arrow"`.
- `write_tables(output_dir, fmt, tables: dict[str, DataFrame]) -> dict[str, Path]`
  / `read_tables(output_dir, fmt, names) -> dict[str, DataFrame]` -- several
  named tables at once, each written to `output_dir/{name}.{ext}`.

Each consuming package keeps its own thin `io.py` with a domain-specific,
typed `write_run(output_dir, fmt, hits, clusters, ...)` /
`read_run(output_dir, fmt) -> (hits, clusters, ...)` built on
`write_tables`/`read_tables` -- callers get the same ergonomic, named-tuple
style API as before; only the serialization internals moved.

## Setup

```bash
cd clustering/utils
python3 -m venv .venv
.venv/bin/pip install -e . -r requirements.txt
```

## Using it from another `clustering/` package

```bash
# from e.g. clustering/sensor/, alongside its own -e .
.venv/bin/pip install -e ../utils -e . -r requirements.txt
```

```python
from clustering_utils.io import write_tables, read_tables

def write_run(output_dir, fmt, hits, clusters, truth):
    return write_tables(output_dir, fmt, {"hits": hits, "clusters": clusters, "truth": truth})

def read_run(output_dir, fmt):
    tables = read_tables(output_dir, fmt, ["hits", "clusters", "truth"])
    return tables["hits"], tables["clusters"], tables["truth"]
```

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers single-table round-trip (CSV/Arrow), parent-directory creation,
unknown-format errors, and multi-table `write_tables`/`read_tables`
round-trip.
