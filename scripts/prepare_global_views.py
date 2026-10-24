"""Generate only the new LGQC caches. Local APPM/T-Net remains external."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfgc.views import prepare_cloud


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--names', type=Path, required=True,
                        help='User-prepared CSV with a Ply_name column; references excluded')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    import pandas as pd
    raw, output = args.raw_root.resolve(), args.output.resolve()
    code = Path(__file__).resolve().parents[1]
    if not raw.is_dir() or output == raw or raw in output.parents or output in raw.parents:
        raise ValueError('Use existing raw input and separate cache output directories')
    if output == code or output in code.parents or code in output.parents:
        raise ValueError('Store generated caches outside the release source directory')
    names = pd.read_csv(args.names).Ply_name.astype(str).str.removesuffix('.ply').tolist()
    if not names or len(set(names)) != len(names):
        raise ValueError('Names CSV must contain a unique nonempty distorted-sample inventory')
    paths = {}
    for path in sorted(raw.rglob('*.ply')):
        if path.stem in paths:
            raise ValueError(f'Duplicate raw name: {path.stem}')
        paths[path.stem] = path
    for index, name in enumerate(names, 1):
        if '/' in name or '\\' in name or name in {'.', '..', ''}:
            raise ValueError('Invalid sample stem')
        if name not in paths:
            raise FileNotFoundError(f'Missing raw distorted sample: {name}')
        if not args.dry_run:
            prepare_cloud(paths[name], output / name)
        print(f'[{index}/{len(names)}] {name}', flush=True)


if __name__ == '__main__':
    main()
