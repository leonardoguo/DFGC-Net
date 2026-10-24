"""MOS-free orchestration using the user's external T-Net and alignment loss.

No T-Net architecture or alignment-loss implementation is redistributed here.
"""
import hashlib
import importlib
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from .runtime import read_json, write_json, sha256, upstream_fingerprint
from .preparation import file_signature, guarded_stage


def stable_seed(text):
    return int(hashlib.sha256(str(text).encode()).hexdigest()[:8], 16)


class NormalPatches(Dataset):
    def __init__(self, cache, records, training):
        self.cache, self.records, self.training = Path(cache), records, training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        index, epoch = index if isinstance(index, tuple) else (index, 0)
        name, patch = self.records[index]
        array = np.load(self.cache / f'{name}.npy', mmap_mode='r', allow_pickle=False)
        normals = torch.from_numpy(np.array(array[patch], copy=True)).float().T.contiguous()
        tag = f'train:{epoch}' if self.training else 'monitor:fixed'
        generator = torch.Generator().manual_seed(stable_seed(f'tnet-rotation:42:{name}:{patch}:{tag}'))
        quaternion_to_matrix = importlib.import_module('model.adaptProjection').quaternion_to_matrix
        rotation = quaternion_to_matrix(torch.randn(1, 4, generator=generator))[0]
        return rotation @ normals


class EpochOrder(Sampler):
    def __init__(self, size):
        self.size, self.epoch = size, 0

    def __len__(self):
        return self.size

    def __iter__(self):
        generator = torch.Generator().manual_seed(42 + self.epoch)
        return iter((int(index), self.epoch) for index in torch.randperm(self.size, generator=generator))


def prepare_normal_cache(data_root, names, upstream, device):
    from pytorch3d.ops import estimate_pointcloud_normals
    import pytorch3d
    helpers = importlib.import_module('data.DatasetUtils')
    models = importlib.import_module('model.IQA_Model')
    patches = {name: [Path(data_root) / 'patches' / f'{name}_part{k}.ply' for k in range(32)] for name in names}
    signature = dict(neighbors=50, points=8192, pytorch3d_version=pytorch3d.__version__,
                     base_model_sha256=sha256(Path(upstream) / 'model/IQA_Model.py'),
                     patch_stats={p.name: file_signature(p) for files in patches.values() for p in files})
    cache = Path(data_root) / 'tnet_normals_k50'
    record = guarded_stage(data_root, 'tnet_normals', signature, cache)
    with torch.no_grad():
        for index, (name, files) in enumerate(patches.items(), 1):
            path = cache / f'{name}.npy'
            if path.is_file():
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                if array.shape != (32, 8192, 3) or not np.isfinite(array).all():
                    raise ValueError(f'Invalid cached normals: {path}')
                continue
            chunks = []
            for start in range(0, 32, 4):
                array = np.stack([helpers.load_ply(str(p))[:, :3] for p in files[start:start+4]])
                points = torch.tensor(array, dtype=torch.float32, device=device)
                normals = estimate_pointcloud_normals(points, neighborhood_size=50)
                normals = models.orient_normals_outward(points, normals)
                if not torch.isfinite(normals).all() or (normals.norm(dim=-1) < 1e-6).any():
                    raise FloatingPointError(f'Non-finite normals: {name}')
                chunks.append(normals.cpu().numpy())
            temporary = path.with_suffix('.npy.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, np.concatenate(chunks).astype(np.float32), allow_pickle=False)
            temporary.replace(path)
            print(f'[normals {index}/{len(names)}] {name}', flush=True)
    write_json(record, dict(signature=signature, status='completed'))
    return cache, signature


def train_tnet(data_root, names, upstream, args):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; install matching PyTorch/PyTorch3D')
    names = sorted(names)
    cache, normal_signature = prepare_normal_cache(data_root, names, upstream, device)
    cfg = dict(seed=42, max_epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
               workers=args.workers, min_delta=1e-4, rotation_mode='quaternion',
               normal_signature=normal_signature, mode='all_unlabelled_28_train_4_fixed_monitor_patches',
               upstream_sha256=upstream_fingerprint(upstream),
               runner_sha256=sha256(__file__))
    output = Path(data_root) / 'tnet'
    output.mkdir(parents=True, exist_ok=True)
    config_file = output / 'config.json'
    if config_file.exists() and read_json(config_file) != cfg:
        raise ValueError('Changed T-Net recipe: use a fresh data root')
    write_json(config_file, cfg)
    train_records, monitor_records = [], []
    for name in names:
        order = np.random.default_rng(stable_seed(f'tnet-split:42:{name}')).permutation(32)
        monitor_records.extend((name, int(k)) for k in sorted(order[:4]))
        train_records.extend((name, int(k)) for k in sorted(order[4:]))
    train = NormalPatches(cache, train_records, True)
    monitor = NormalPatches(cache, monitor_records, False)
    sampler = EpochOrder(len(train))
    extras = dict(persistent_workers=True, prefetch_factor=2) if args.workers else {}
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, **extras)
    monitor_loader = DataLoader(monitor, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, **extras)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    BaseTNet = importlib.import_module('model.IQA_Model').TNetAlignmentModel
    external = BaseTNet(learning_rate=5e-5, weight_decay=1e-5, rotation_mode='quaternion').to(device)
    optimizer = torch.optim.Adam(external.stn.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.97)
    start, best, stale, best_epoch, history = 0, float('inf'), 0, -1, []
    last = output / 'last.ckpt'
    if last.exists():
        saved = torch.load(last, map_location=device)
        if saved.get('finished'):
            print(f'[SKIP] T-Net finished; best={output / "best.ckpt"}')
            return
        external.stn.load_state_dict({k[4:]: v for k,v in saved['state_dict'].items()}, strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start, best, stale = saved['epoch'] + 1, saved['best_val'], saved['stale']
        best_epoch, history = saved['best_epoch'], saved['history']
        torch.set_rng_state(saved['rng_cpu'].cpu())
        if device.type == 'cuda' and saved['rng_cuda'] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in saved['rng_cuda']])

    def loop(loader, training):
        external.train(training)
        total, count = 0., 0
        with torch.set_grad_enabled(training):
            for normals in loader:
                normals = normals.to(device)
                transform = external.stn(normals)
                # Alignment loss and T-Net come from the separate upstream model.
                aligned = torch.bmm(transform.float(), normals.float())
                loss = external.normal_loss(aligned, normals.new_tensor([0., 0., 1.])).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite T-Net loss')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total += loss.item() * len(normals)
                count += len(normals)
        return total / count

    import pandas as pd
    for epoch in range(start, args.epochs):
        sampler.epoch = epoch
        training_loss, validation_loss = loop(train_loader, True), loop(monitor_loader, False)
        scheduler.step()
        improved = validation_loss < best - 1e-4
        if improved:
            best, stale, best_epoch = validation_loss, 0, epoch
        else:
            stale += 1
        history.append(dict(epoch=epoch, train_loss=training_loss, val_loss=validation_loss,
                            best_val=best, best_epoch=best_epoch))
        payload = dict(state_dict={'stn.' + key: value.detach().cpu() for key,value in external.stn.state_dict().items()},
                       hyper_parameters={'rotation_mode': 'quaternion'}, epoch=epoch, best_val=best,
                       best_epoch=best_epoch, stale=stale, history=history,
                       optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       rng_cpu=torch.get_rng_state(),
                       rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                       finished=stale >= args.patience or epoch + 1 == args.epochs)
        for destination in ([output / 'best.ckpt', last] if improved else [last]):
            temporary = destination.with_suffix('.ckpt.tmp')
            torch.save(payload, temporary)
            temporary.replace(destination)
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        print(f'[tnet] epoch={epoch+1}/{args.epochs} train={training_loss:.6f} val={validation_loss:.6f} '
              f'best_epoch={best_epoch+1} stale={stale}/{args.patience}', flush=True)
        if payload['finished']:
            break
    print(f'[tnet] use {output / "best.ckpt"} for local PNG generation (not last.ckpt)')
