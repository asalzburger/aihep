"""The exchange point: name -> Splitter class. Add a new splitter by
writing a `base.Splitter` subclass and adding one entry here -- the CLI and
`cli.run_split` pick it up automatically via `--splitter <name>`.
"""

from __future__ import annotations

from .base import Splitter
from .truth_splitter import TruthSplitter

SPLITTERS: dict[str, type[Splitter]] = {
    TruthSplitter.name: TruthSplitter,
}
