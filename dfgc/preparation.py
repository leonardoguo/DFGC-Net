"""Our preparation orchestration; local algorithms are loaded from user DA-Net."""
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import numpy as np

from .runtime import ROOT, read_json, write_json, sha256


def prepare_metadata(dataset, data_root, upstream, source=None):
    import pandas as pd
    root = Path(data_root)
    recipe = read_json(ROOT / 'configs/splits' / f'{dataset}.json')
    expected = set(recipe['1']['train']) | set(recipe['1']['vail'])
    if source:
        source = Path(source)
        frame = pd.read_csv(source) if source.suffix.lower() == '.csv' else pd.read_excel(source)
    elif dataset == 'wpc':
        folder = Path(upstream) / 'configs/wpc/1'
        frame = pd.concat([pd.read_excel(folder / 'train.xlsx'), pd.read_excel(folder / 'vail.xlsx')],
                          ignore_index=True)
    else:
        from scipy.io import loadmat
        from .sjtu_factors import CONTENTS
        source = root / 'Final_MOS.mat'
        if not source.is_file():
            raise FileNotFoundError(f'Keep the official Final_MOS.mat at {source}; or supply normalized --metadata CSV')
        values = np.asarray(loadmat(source)['Final_MOS'])
        if values.shape != (42, 9):
            raise ValueError('Expected official SJTU Final_MOS matrix [42,9]')
        # The official readme gives this column order; UBL_unicorn is the readme
        # spelling, while actual PLYs use "ULB Unicorn". Do not alphabetize it.
        frame = pd.DataFrame([dict(Ply_name=f'{name}_{index}.ply', Content=name, MOS=float(values[index, col]))
                              for col, name in enumerate(CONTENTS) for index in range(42)])
    required = ['Ply_name', 'Content', 'MOS']
    if dataset == 'wpc':
        required += ['Distortion_Type', 'G_Parameter', 'T_Parameter']
    if not set(required).issubset(frame.columns):
        raise ValueError(f'Metadata must have {required}; do not use a headerless MOS.xlsx as a named table')
    frame = frame[required].copy()
    frame['Ply_name'] = frame.Ply_name.astype(str)
    if frame.Ply_name.duplicated().any() or set(frame.Ply_name) != expected:
        raise ValueError('Metadata sample names must exactly cover the released distorted-sample inventory')
    if not np.isfinite(frame.MOS.to_numpy(dtype=float)).all():
        raise ValueError('Invalid MOS')
    maximum = 100 if dataset == 'wpc' else 10
    if (frame.MOS < 0).any() or (frame.MOS > maximum).any():
        raise ValueError(f'{dataset} MOS must be on original 0..{maximum} scale, not pre-divided')
    root.mkdir(parents=True, exist_ok=True)
    csv = root / 'metadata.csv'
    if csv.exists():
        old = pd.read_csv(csv)
        pd.testing.assert_frame_equal(old.sort_values('Ply_name').reset_index(drop=True),
                                      frame.sort_values('Ply_name').reset_index(drop=True), check_dtype=False,
                                      check_exact=False, rtol=1e-12, atol=1e-12)
    else:
        frame.to_csv(csv, index=False)
    indexed = frame.set_index('Ply_name', drop=False)
    for fold, partitions in recipe.items():
        tables = [indexed.loc[partitions[key]].reset_index(drop=True) for key in ('train', 'vail')]
        if set(tables[0].Content) & set(tables[1].Content):
            raise ValueError(f'Content overlap in fold {fold}; source metadata is inconsistent')
        for key, table in zip(('train', 'vail'), tables):
            path = root / 'splits' / f'fold_{fold}' / f'{key}.xlsx'
            if path.exists():
                pd.testing.assert_frame_equal(pd.read_excel(path), table, check_dtype=False,
                                              check_exact=False, rtol=1e-12, atol=1e-12)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                table.to_excel(path, index=False)
    print(f'[metadata] {dataset}: {len(frame)} samples; {len(recipe)} content-disjoint folds -> {root}')
    return frame


def raw_inventory(data_root, dataset, names):
    root = Path(data_root) / ('raw' if dataset == 'wpc' else 'distortion')
    paths = {}
    for path in sorted(root.rglob('*.ply')):
        name = path.stem
        # Original WPC distributions sometimes append _rounded to 80 files.
        # Alias at lookup only; never rename the user's raw dataset in place.
        if dataset == 'wpc' and name.endswith('_rounded'):
            name = name[:-8]
        if name in paths:
            raise ValueError(f'Duplicate raw alias {name}; keep one source file')
        paths[name] = path
    if set(names) - paths.keys():
        raise FileNotFoundError(f'Raw distorted PLYs missing from {root}: {sorted(set(names)-paths.keys())[:6]}')
    return [(name, paths[name]) for name in names]


def guarded_stage(data_root, stage, signature, output):
    """Do not silently overwrite imported caches or mix different recipes."""
    record = Path(data_root) / 'preprocessing' / f'{stage}.json'
    output = Path(output)
    if record.is_file():
        if read_json(record)['signature'] != signature:
            raise ValueError(f'{stage} recipe changed; use a new data root/output, not mixed caches')
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f'Existing {output} has no generation record. Use it directly for training if compatible; '
                         'generate new assets in a fresh data root instead of overwriting imported caches.')
    write_json(record, dict(signature=signature, status='running'))
    output.mkdir(parents=True, exist_ok=True)
    return record


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def prepare_patches(data_root, dataset, names, workers=1):
    import torch
    process = importlib.import_module('utils.PCUtils').process
    records = raw_inventory(data_root, dataset, names)
    output = Path(data_root) / 'patches'
    signature = dict(algorithm='external PCUtils.process', points=8192, patches=32, seed=42,
                     source_sha256=sha256(Path(importlib.import_module('utils.PCUtils').__file__)),
                     raw={name: file_signature(path) for name, path in records})
    record = guarded_stage(data_root, 'patches', signature, output)
    # Serial calls give each sample a fixed seed, independent of pool scheduling.
    # The workers flag applies to training/normal loading, not this FPS loop.
    for index, (name, source) in enumerate(records, 1):
        marker = Path(data_root) / 'preprocessing/patch_done' / f'{name}.json'
        files = [output / f'{name}_part{k}.ply' for k in range(32)]
        if marker.is_file() and all(p.is_file() and p.stat().st_size > 0 for p in files):
            continue
        seed = int(hashlib.sha256(f'patches:42:{name}'.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        np.random.seed(seed)
        result = process(str(source), str(output), 8192, 32, name)
        if not result[1] or not all(p.is_file() and p.stat().st_size > 0 for p in files):
            raise RuntimeError(f'External patch generation failed: {result}')
        write_json(marker, dict(completed=True))
        print(f'[patches {index}/{len(records)}] {name}', flush=True)
    write_json(record, dict(signature=signature, status='completed'))


def prepare_views(data_root, dataset, names):
    from .views import prepare_cloud
    for index, (name, source) in enumerate(raw_inventory(data_root, dataset, names), 1):
        prepare_cloud(source, Path(data_root) / 'global_views' / name)
        print(f'[views {index}/{len(names)}] {name}', flush=True)


def upstream_main(upstream):
    name = '_dfgc_external_main'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(upstream) / 'main.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def prepare_pngs(data_root, names, upstream, checkpoint, device):
    import torch
    from PIL import Image
    api = upstream_main(upstream)
    read_ply = importlib.import_module('data.DatasetUtils').load_ply
    output = Path(data_root) / 'projections'
    patches = [Path(data_root) / 'patches' / f'{name}_part{k}.ply' for name in names for k in range(32)]
    signature = dict(tnet_sha256=sha256(checkpoint), main_sha256=sha256(Path(upstream) / 'main.py'),
                     projection_sha256=sha256(Path(upstream) / 'utils/projection.py'),
                     backend='pytorch3d_scatter', patches={p.name: file_signature(p) for p in patches})
    record = guarded_stage(data_root, 'png', signature, output)
    payload = torch.load(checkpoint, map_location='cpu')
    mode = payload.get('hyper_parameters', {}).get('rotation_mode', 'quaternion')
    stn = api.load_stn_from_checkpoint(str(checkpoint), torch.device(device), rotation_mode=mode)
    stn.eval().requires_grad_(False)
    renderer = api.build_pytorch3d_scatter_renderer(torch.device(device))
    with torch.no_grad():
        for index, path in enumerate(patches, 1):
            target = output / f'{path.stem}_1.png'
            if target.is_file():
                with Image.open(target) as image:
                    if image.size != (224, 224) or image.mode != 'RGB':
                        raise ValueError(f'Invalid existing PNG: {target}')
                    image.verify()
                continue
            patch = torch.tensor(read_ply(str(path)), dtype=torch.float32)
            if tuple(patch.shape) != (8192, 6) or not torch.isfinite(patch).all():
                raise ValueError(f'Invalid patch {path}')
            image = api.generate_projection_image(renderer, stn, patch, 'pytorch3d_scatter')
            if image.mode != 'RGB' or image.size != (224, 224):
                raise ValueError('External renderer must return RGB 224x224')
            temporary = target.with_suffix('.png.tmp')
            image.save(temporary, format='PNG')
            temporary.replace(target)
            if index % 32 == 0:
                print(f'[png] {index}/{len(patches)}', flush=True)
    write_json(record, dict(signature=signature, status='completed'))
