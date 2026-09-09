"""Responses to dataset: connectivity, k-core, stratified sampling, holdout, charts.

Only the container type is re-exported. `build`, `save` and `load` stay behind
`dataset.` because `data.load(...)` would read as ambiguously as `config.load(...)`,
and `allocate` or `cell_mask` alone does not say what it acts on.
"""

from .dataset import Dataset

__all__ = ["Dataset"]
