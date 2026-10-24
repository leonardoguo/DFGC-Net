"""Train DFGC-Net by importing a separately obtained, unmodified DA-Net."""
import argparse
import gc
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfgc.runtime import (ROOT, bind_upstream, upstream_fingerprint, configuration, check_split,
                          check_assets, build_datasets, loader, make_model, read_json, write_json,
                          sha256, metric_values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--dataset', choices=['wpc', 'sjtu'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--swin', type=Path)
    parser.add_argument('--split-root', type=Path, help='Optional existing split directory; default DATA_ROOT/splits')
    parser.add_argument('--folds', nargs='+', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-batches', type=int, default=4,
                        help='Batches per loop in smoke mode; increase if initial AMP updates are skipped')
    args = parser.parse_args()
    upstream = bind_upstream(args.upstream)
    if args.workers is not None and args.workers < 0 or args.epochs is not None and args.epochs < 1:
        parser.error('workers >= 0 and epochs >= 1 required')
    if args.smoke_batches < 1:
        parser.error('smoke-batches must be positive')
    cfg0 = configuration(args.dataset, args.data_root, 1, args.swin, args.split_root)
    folds = list(dict.fromkeys(args.folds or cfg0['folds']))
    output = (args.output or ROOT / 'outputs' / args.dataset).resolve()
    for fold in folds:
        cfg = configuration(args.dataset, args.data_root, fold, args.swin, args.split_root)
        if args.workers is not None:
            cfg['hyperparameters']['num_workers'] = args.workers
        if args.epochs is not None:
            cfg['hyperparameters']['num_epochs'] = args.epochs
        if args.smoke:
            cfg['hyperparameters']['num_workers'] = 0
        frames = check_split(cfg)
        count = check_assets(cfg, frames)
        if not Path(cfg['pretrained']).is_file():
            raise FileNotFoundError(f'Place Swin-T ImageNet-1K weights at {cfg["pretrained"]} or supply --swin')
        cfg['runtime'] = dict(upstream=str(upstream), upstream_sha256=upstream_fingerprint(upstream),
                              extension_sha256={p.name: sha256(p) for p in sorted((ROOT / 'dfgc').glob('*.py'))},
                              split_sha256={key: sha256(cfg['dataset'][key]) for key in ('traininfo', 'vailinfo')},
                              swin_sha256=sha256(cfg['pretrained']), device=args.device,
                              train_entry_sha256=sha256(__file__))
        if args.smoke:
            cfg['runtime']['smoke_batches'] = args.smoke_batches
        if args.dry_run:
            print(f'[PASS] {args.dataset} fold {fold}: {count} clouds, all caches/config/paths; no training')
            continue
        import torch
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.loggers import TensorBoardLogger
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; explicitly use --device cpu only for CPU checks')
        run = output / ('_smoke' if args.smoke else '') / f'fold_{fold}' / 'version_0'
        if (run / 'config.json').exists() and read_json(run / 'config.json') != cfg:
            raise ValueError('Changed configuration/source; use a fresh --output directory')
        if (run / 'completed.json').exists() and not args.smoke:
            if not Path(read_json(run / 'completed.json')['checkpoint']).is_file():
                raise FileNotFoundError('Completed run checkpoint missing')
            print(f'[SKIP] completed fold {fold}')
            continue
        pl.seed_everything(cfg['torch']['seed'])
        torch.set_float32_matmul_precision(cfg['torch']['precision'])
        train, val = build_datasets(cfg)
        train_loader, val_loader = loader(train, cfg, True), loader(val, cfg, False)
        model = make_model(cfg)
        write_json(run / 'config.json', cfg)
        checkpoint = ModelCheckpoint(dirpath=str(run / 'checkpoints'), monitor='acc/val_srocc', mode='max',
                                     save_top_k=3, save_last=True, filename='epoch={epoch:03d}',
                                     auto_insert_metric_name=False)
        logger = TensorBoardLogger(str(run.parent.parent), name=f'fold_{fold}', version=0)
        trainer = pl.Trainer(accelerator='gpu' if args.device == 'cuda' else 'cpu', devices=1,
                             max_epochs=cfg['hyperparameters']['num_epochs'],
                             precision='16-mixed' if args.device == 'cuda' else 32,
                             callbacks=[checkpoint], logger=logger,
                             fast_dev_run=args.smoke_batches if args.smoke else False,
                             log_every_n_steps=10, check_val_every_n_epoch=1)
        last = run / 'checkpoints/last.ckpt'
        trainer.fit(model, train_loader, val_loader, ckpt_path=str(last) if last.exists() and not args.smoke else None)
        if args.smoke:
            states = [state for optimizer in trainer.optimizers for state in optimizer.state.values()]
            updates = max((int(state['step']) for state in states if 'step' in state), default=0)
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            finite_gradients = all(torch.isfinite(g).all().item() for g in gradients) if gradients else None
            finite_model = all(torch.isfinite(value).all().item() for value in model.state_dict().values())
            finite_optimizer = all(torch.isfinite(value).all().item() for state in states
                                   for value in state.values() if isinstance(value, torch.Tensor))
            report = dict(dataset=args.dataset, fold=fold, device=args.device,
                          parameters=sum(p.numel() for p in model.parameters()),
                          backbone_2d=type(model.pic_backbone).__name__, backbone_3d=type(model.pc_backbone).__name__,
                          requested_batches=args.smoke_batches, optimizer_updates=updates,
                          observed_gradient_tensors=len(gradients), finite_final_gradients=finite_gradients,
                          finite_model_state=finite_model, finite_optimizer_state=finite_optimizer,
                          metrics={key: float(value) for key, value in trainer.callback_metrics.items()
                                   if getattr(value, 'numel', lambda: 0)() == 1},
                          passed=updates > 0 and finite_model and finite_optimizer)
            write_json(run / 'smoke_report.json', report)
            print(f'[smoke] {report}', flush=True)
            if not report['passed']:
                raise RuntimeError('Smoke forward finished but no finite optimizer update was verified. '
                                   'If initial AMP scaling skipped updates, retry in a fresh --output with '
                                   '--smoke-batches 4 (or more). See smoke_report.json.')
        if not args.smoke:
            import pandas as pd
            best = Path(checkpoint.best_model_path)
            epoch = int(re.fullmatch(r'epoch=(\d+)(?:-v\d+)?\.ckpt', best.name).group(1))
            result_path = run / 'predictions' / f'epoch_{epoch:03d}.csv'
            frame = pd.read_csv(result_path)
            write_json(run / 'completed.json', dict(fold=fold, best_epoch=epoch, checkpoint=str(best),
                       predictions=str(result_path), **metric_values(frame.mos, frame.pred)))
        del trainer, model, train_loader, val_loader, train, val
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.smoke:
            break
    if not args.dry_run and not args.smoke:
        import pandas as pd
        records = [read_json(output / f'fold_{fold}/version_0/completed.json') for fold in cfg0['folds']
                   if (output / f'fold_{fold}/version_0/completed.json').is_file()]
        if records:
            frame = pd.DataFrame(records)
            keys = ['srocc', 'plcc_raw', 'plcc_5param', 'krocc', 'rmse_5param']
            mean = dict(fold='mean_all_folds' if len(records) == len(cfg0['folds']) else 'mean_available',
                        **frame[keys].mean().to_dict())
            pd.concat([frame, pd.DataFrame([mean])], ignore_index=True).to_csv(output / 'summary.csv', index=False)


if __name__ == '__main__':
    main()
