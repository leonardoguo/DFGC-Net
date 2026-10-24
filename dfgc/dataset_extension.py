"""Add only GTFS metadata and LGQC mosaics to a user-supplied paired dataset."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .views import stable_seed, cache_complete, load_views, build_mosaic


class ContextDataset(Dataset):
    """Base returns (name_or_metadata, pcs, pics, scaled_mos).

    `names` must match base indexing exactly, not a separately sorted inventory.
    `factors` maps every name to the four GTFS label/mask fields.
    This wrapper never loads/normalizes/rotates local patches or paired PNGs.
    """
    def __init__(self, base, names, factors, cache_root, training, seed=42):
        if len(base) != len(names) or len(set(names)) != len(names):
            raise ValueError("Unique names in base-dataset order are required")
        self.base, self.names, self.factors = base, tuple(names), factors
        self.cache_root, self.training, self.seed = Path(cache_root), bool(training), int(seed)
        self._rng, self._worker_seed = None, None
        for name in names:
            if Path(name).name != name or '/' in name or '\\' in name or name in {'.', '..', ''}:
                raise ValueError("Expected sample stem, not a path")
            if name not in factors:
                raise KeyError(f"Missing factor metadata for {name}")
            if not cache_complete(self.cache_root / name):
                raise FileNotFoundError(f"Missing six-view cache: {name}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        source, pcs, pics, mos = self.base[index]
        name = self.names[index]
        if isinstance(source, dict):
            metadata = dict(source)
            if 'name' in metadata and metadata['name'] != name:
                raise ValueError("Base names/order differ from supplied names")
        else:
            if str(source).removesuffix('.ply') != name:
                raise ValueError("Base names/order differ from supplied names")
            metadata = {}
        metadata.update(self.factors[name])
        metadata['name'] = name
        if self.training:
            worker_seed = torch.initial_seed()
            if self._rng is None or self._worker_seed != worker_seed:
                self._worker_seed = worker_seed
                self._rng = np.random.default_rng(stable_seed(f'{self.seed}:train:{worker_seed}'))
            rng = self._rng
        else:
            rng = np.random.default_rng(stable_seed(f'{self.seed}:validation:{name}'))
        image = build_mosaic(*load_views(self.cache_root / name), rng)
        metadata['global_mosaic'] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        return metadata, pcs, pics, mos
