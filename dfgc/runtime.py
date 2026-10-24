"""Reader-facing orchestration. The base model/loader/metrics stay external."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bind_upstream(path):
    """Select one user-obtained checkout per process; never copy/patch it."""
    root = Path(path).resolve()
    if root == ROOT or ROOT in root.parents:
        raise ValueError('Keep DA-Net outside DFGC-Net, e.g. a sibling DA-Net directory')
    for file in ('model/IQA_Model.py', 'model/metrics.py', 'data/Datasets.py', 'utils/PCUtils.py'):
        if not (root / file).is_file():
            raise FileNotFoundError(f'Missing {root / file}; --upstream must name the actual DA-Net root')
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in {'model', 'data', 'utils'}:
            source = getattr(module, '__file__', None)
            if source and root not in Path(source).resolve().parents:
                raise RuntimeError(f'Conflicting imported {name}; start a fresh process')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Spawned Windows workers need the same external checkout on their path.
    return root


def upstream_fingerprint(root):
    files = []
    for directory in ('model', 'data', 'utils'):
        files.extend((Path(root) / directory).rglob('*.py'))
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files)}


def configuration(dataset, data_root, fold, swin=None, split_root=None):
    cfg = read_json(ROOT / 'configs' / f'{dataset}.json')
    if fold not in cfg['folds']:
        raise ValueError(f'Invalid {dataset} fold {fold}')
    data_root = Path(data_root).resolve()
    for key in ('pc_path', 'pic_path', 'global_view_cache', 'raw_path'):
        cfg['dataset'][key] = str(data_root / cfg['dataset'][key])
    for key, partition in [('traininfo', 'train'), ('vailinfo', 'vail')]:
        cfg['dataset'][key] = str((Path(split_root).resolve() if split_root else data_root / 'splits')
                                   / f'fold_{fold}' / f'{partition}.xlsx')
    cfg['pretrained'] = str(Path(swin).resolve() if swin else ROOT / cfg['pretrained'])
    cfg['fold'], cfg['data_root'] = fold, str(data_root)
    return cfg


def check_split(cfg):
    import numpy as np
    import pandas as pd
    dataset, fold = cfg['dataset']['dataset_name'], cfg['fold']
    recipe = read_json(ROOT / 'configs/splits' / f'{dataset}.json')[str(fold)]
    frames = []
    for key, partition in [('traininfo', 'train'), ('vailinfo', 'vail')]:
        frame = pd.read_excel(cfg['dataset'][key])
        if frame.Ply_name.astype(str).tolist() != recipe[partition]:
            raise ValueError(f'Changed sample order/membership: {key}; run prepare --stage metadata')
        if not np.isfinite(frame.MOS.to_numpy(dtype=float)).all():
            raise ValueError('Non-finite MOS')
        frames.append(frame)
    if set(frames[0].Content) & set(frames[1].Content):
        raise ValueError('Content overlap in train/held-out split')
    expected = (592, 148) if dataset == 'wpc' else (336, 42)
    if tuple(map(len, frames)) != expected:
        raise ValueError('Unexpected split sizes')
    return frames


def check_assets(cfg, frames=None):
    from .views import cache_complete
    frames = frames or check_split(cfg)
    missing, ds = [], cfg['dataset']
    names = [str(name).removesuffix('.ply') for frame in frames for name in frame.Ply_name]
    for name in names:
        for index in range(32):
            for directory, tail in [('pc_path', f'{name}_part{index}.ply'),
                                    ('pic_path', f'{name}_part{index}_1.png')]:
                path = Path(ds[directory]) / tail
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(str(path))
        if not cache_complete(Path(ds['global_view_cache']) / name):
            missing.append(f'global_views/{name}')
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing cache entries; first: {missing[:6]}')
    return len(names)


def build_datasets(cfg):
    """Delegate ALL local loading/normalization/augmentation to upstream."""
    from .dataset_extension import ContextDataset
    from .sjtu_factors import factor_metadata as sjtu_labels
    from .wpc_factors import factor_metadata as wpc_labels
    BaseDataset = importlib.import_module('data.Datasets').WPC10MultiDataset
    ds, result = cfg['dataset'], []
    for training, frame in zip((True, False), check_split(cfg)):
        base = BaseDataset(pc_path=ds['pc_path'], pic_path=ds['pic_path'],
                           mos_path=ds['traininfo' if training else 'vailinfo'], patch_num=32,
                           select_patch=8 if training else 32,
                           mos_scale=ds['train_mos_scale' if training else 'vail_mos_scale'],
                           random_select=training, data_aug=bool(training and ds['data_aug']),
                           data_normalization=ds['data_normalization'])
        names = [str(name).removesuffix('.ply') for name in frame.Ply_name]
        if ds['dataset_name'] == 'sjtu':
            factors = {name: sjtu_labels(name) for name in names}
        else:
            factors = {str(row.Ply_name).removesuffix('.ply'):
                       wpc_labels(str(row.Distortion_Type), row.G_Parameter, row.T_Parameter)
                       for row in frame.itertuples(index=False)}
        result.append(ContextDataset(base, names, factors, ds['global_view_cache'], training,
                                     cfg['torch']['seed']))
    return result


def loader(dataset, cfg, training):
    from torch.utils.data import DataLoader
    workers = cfg['hyperparameters']['num_workers']
    extra = dict(persistent_workers=True, prefetch_factor=2) if workers > 0 else {}
    return DataLoader(dataset, batch_size=cfg['hyperparameters']['batch_size'] if training else 1,
                      shuffle=bool(training), num_workers=workers,
                      pin_memory=cfg['hyperparameters']['pin_memory'], **extra)


def metric_values(mos, prediction):
    import numpy as np
    from scipy.stats import pearsonr
    calculate = importlib.import_module('model.metrics').IQA_accruacy
    y, p = np.asarray(mos, dtype=np.float32), np.asarray(prediction, dtype=np.float32)
    s, pl, k, rm = calculate(y, p)
    result = dict(srocc=float(s), plcc_raw=float(pearsonr(y, p).statistic),
                  plcc_5param=float(pl), krocc=float(k), rmse_5param=float(rm))
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError('Non-finite metrics: check constant predictions, MOS and preprocessing')
    return result


def make_model(cfg):
    """Build a runtime subclass, not a redistributed copy of the host forward."""
    import torch
    from .integration import DFGCExtensions
    BaseModel = importlib.import_module('model.IQA_Model').MultiModel

    class ReaderModel(BaseModel):
        def __init__(self):
            hp = cfg['hyperparameters']
            keys = ('embedding_size', 'embedding_dim', 'l_w_embedding', 'l_w_commitment')
            super().__init__(optimizer_param=hp['optimizer_param'], swin_ckpt_path=cfg['pretrained'],
                             **{key: hp['model'][key] for key in keys})
            self.dfgc_extension = DFGCExtensions(
                self, geometry_weight=hp['model']['gtfs_geometry_loss_weight'],
                texture_weight=hp['model']['gtfs_texture_loss_weight'],
                beta=hp['model']['gtfs_smooth_l1_beta'], hidden_dim=hp['model']['gtfs_hidden_dim'],
                preserve_cuda_rng=cfg['dataset']['dataset_name'] == 'sjtu')
            self._reader_rows = []

        def _reader_forward(self, batch, training):
            metadata, pcs, pics, mos = batch
            if not all(torch.isfinite(value).all() for value in (pcs, pics, mos, metadata['global_mosaic'])):
                raise FloatingPointError('Non-finite input; check data normalization and preprocessing')
            if cfg['dataset']['dataset_name'] == 'sjtu' and pcs.abs().max() > 1.001:
                raise ValueError('SJTU local XYZRGB must be normalized by the upstream loader')
            with self.dfgc_extension.batch(metadata['global_mosaic'], pcs.shape[0], pcs.shape[1],
                                           epoch=self.current_epoch, step=self.global_step) as context:
                prediction, base = self(pcs, pics, mos)
                loss = base['loss']
                if training:
                    auxiliary, terms = context.auxiliary_loss(metadata)
                    loss = loss + auxiliary
                    for key in ('geometry_loss', 'texture_loss'):
                        self.log(f'gtfs/{key}', terms[key], on_step=False, on_epoch=True,
                                 batch_size=mos.numel())
            if not torch.isfinite(loss).all() or not torch.isfinite(prediction).all():
                raise FloatingPointError('Non-finite prediction/loss; no checkpoint will be reported')
            self.log('loss/train_loss' if training else 'loss/val_loss', loss,
                     on_step=False, on_epoch=True, prog_bar=True, batch_size=mos.numel())
            return prediction, loss

        def training_step(self, batch, batch_idx):
            return self._reader_forward(batch, True)[1]

        def on_validation_epoch_start(self):
            self._reader_rows.clear()

        def validation_step(self, batch, batch_idx):
            prediction, loss = self._reader_forward(batch, False)
            self._reader_rows.extend(zip(batch[0]['name'], prediction.detach().float().view(-1).cpu().tolist(),
                                         batch[3].detach().float().view(-1).cpu().tolist()))
            return loss

        def on_validation_epoch_end(self):
            if self.trainer.sanity_checking or self.trainer.fast_dev_run or len(self._reader_rows) < 5:
                self._reader_rows.clear()
                return
            import pandas as pd
            frame = pd.DataFrame(self._reader_rows, columns=['name', 'pred', 'mos'])
            values = metric_values(frame.mos, frame.pred)
            self.log_dict({f'acc/val_{key}': value for key, value in values.items()},
                          prog_bar=True, on_step=False, on_epoch=True)
            output = Path(self.logger.log_dir) / 'predictions'
            output.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output / f'epoch_{self.current_epoch:03d}.csv', index=False)
            print(f'[validation] epoch={self.current_epoch} {values}', flush=True)
            self._reader_rows.clear()

        # Do not inherit the host's old batch-end log callbacks: our steps return
        # a loss tensor, not the host's private output dict.
        def on_train_batch_end(self, outputs, batch, batch_idx):
            pass

        def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
            pass

        def on_save_checkpoint(self, checkpoint):
            import random
            import numpy as np
            checkpoint['dfgc_rng'] = dict(python=random.getstate(), numpy=np.random.get_state(),
                                          cpu=torch.get_rng_state(),
                                          cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

        def on_load_checkpoint(self, checkpoint):
            import random
            import numpy as np
            if 'dfgc_rng' in checkpoint:
                state = checkpoint['dfgc_rng']
                random.setstate(state['python'])
                np.random.set_state(state['numpy'])
                torch.set_rng_state(state['cpu'].cpu())
                if torch.cuda.is_available() and state['cuda'] is not None:
                    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])

    return ReaderModel()
