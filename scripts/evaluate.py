"""Evaluate a combined checkpoint using a separately obtained DA-Net host."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--dataset', choices=['wpc', 'sjtu'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--swin', type=Path)
    parser.add_argument('--split-root', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if args.workers < 0:
        parser.error('workers must be nonnegative')
    from dfgc.runtime import bind_upstream, configuration, check_assets, build_datasets, loader, make_model, write_json, metric_values
    bind_upstream(args.upstream)
    import torch
    import pandas as pd
    import pytorch_lightning as pl
    cfg = configuration(args.dataset, args.data_root, args.fold, args.swin, args.split_root)
    cfg['hyperparameters']['num_workers'] = args.workers
    check_assets(cfg)
    pl.seed_everything(cfg['torch']['seed'])
    torch.set_float32_matmul_precision(cfg['torch']['precision'])
    _, val = build_datasets(cfg)
    model = make_model(cfg)
    payload = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(payload.get('state_dict', payload), strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    rows = []
    with torch.no_grad():
        for metadata, pcs, pics, mos in loader(val, cfg, False):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                with model.dfgc_extension.batch(metadata['global_mosaic'].to(device), pcs.shape[0], pcs.shape[1]):
                    prediction, _ = model(pcs.to(device), pics.to(device))
            rows.extend(zip(metadata['name'], prediction.float().view(-1).cpu().tolist(), mos.view(-1).tolist()))
    if args.output.exists():
        raise FileExistsError(f'Choose a fresh output CSV: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=['name', 'pred', 'mos'])
    frame.to_csv(args.output, index=False)
    values = metric_values(frame.mos, frame.pred)
    write_json(args.output.with_suffix('.metrics.json'), values)
    print(values)


if __name__ == '__main__':
    main()
