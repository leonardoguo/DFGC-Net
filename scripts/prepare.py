"""Reader workflow: metadata, external local preparation, and our global views."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--dataset', choices=['wpc', 'sjtu'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--stage', choices=['metadata', 'patches', 'views', 'tnet', 'png', 'check'], required=True)
    parser.add_argument('--metadata', type=Path, help='Optional normalized CSV/XLSX, not a headerless score matrix')
    parser.add_argument('--tnet-checkpoint', type=Path)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    from dfgc.runtime import ROOT, bind_upstream, configuration, check_assets
    from dfgc.preparation import prepare_metadata, prepare_patches, prepare_views, prepare_pngs
    upstream = bind_upstream(args.upstream)
    data_root = args.data_root.resolve()
    for protected in (ROOT, upstream):
        if data_root == protected or protected in data_root.parents or data_root in protected.parents:
            raise ValueError('Data must be separate from both source directories; use a sibling data folder')
    if args.workers < 0 or min(args.epochs, args.patience) < 1 or args.batch_size < 2:
        parser.error('workers>=0, epochs/patience>=1, T-Net batch-size>=2 required')
    if args.stage == 'metadata':
        prepare_metadata(args.dataset, data_root, upstream, args.metadata)
        return
    import pandas as pd
    names = pd.read_csv(data_root / 'metadata.csv').Ply_name.astype(str).str.removesuffix('.ply').tolist()
    if args.stage == 'patches':
        prepare_patches(data_root, args.dataset, names, args.workers)
    elif args.stage == 'views':
        prepare_views(data_root, args.dataset, names)
    elif args.stage == 'tnet':
        from dfgc.tnet_runner import train_tnet
        train_tnet(data_root, names, upstream, args)
    elif args.stage == 'png':
        checkpoint = args.tnet_checkpoint or data_root / 'tnet/best.ckpt'
        prepare_pngs(data_root, names, upstream, checkpoint, args.device)
    else:
        count = check_assets(configuration(args.dataset, data_root, 1))
        print(f'[PASS] {count} clouds: 32 local pairs/cloud and complete global caches')


if __name__ == '__main__':
    main()
