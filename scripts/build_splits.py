"""Join release split-name recipes to MOS obtained by the user, not bundled MOS."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['wpc', 'sjtu'], required=True)
    parser.add_argument('--metadata', type=Path, required=True,
                        help='User CSV with Ply_name,Content,MOS and WPC factor parameters')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import pandas as pd
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == root or output in root.parents or root in output.parents:
        raise ValueError('Write generated split tables outside the release source tree')
    frame = pd.read_csv(args.metadata)
    required = {'Ply_name', 'Content', 'MOS'}
    if args.dataset == 'wpc':
        required.update({'Distortion_Type', 'G_Parameter', 'T_Parameter'})
    if not required.issubset(frame.columns) or frame.Ply_name.duplicated().any():
        raise ValueError(f'Expected unique samples and columns {sorted(required)}')
    if not pd.to_numeric(frame.MOS, errors='coerce').map(math.isfinite).all():
        raise ValueError('Invalid MOS; do not fill missing values with predicted/pseudo scores')
    recipes = json.loads((root / 'configs/splits' / f'{args.dataset}.json').read_text('utf-8'))
    indexed = frame.set_index('Ply_name', drop=False)
    planned = []
    for fold, recipe in recipes.items():
        tables = []
        for partition in ('train', 'vail'):
            names = recipe[partition]
            if len(set(names)) != len(names) or set(names) - set(indexed.index):
                raise ValueError(f'Missing or duplicate sample names in fold {fold}/{partition}')
            table = indexed.loc[names].reset_index(drop=True)
            destination = output / f'fold_{fold}' / f'{partition}.xlsx'
            if destination.exists():
                raise FileExistsError(f'Will not overwrite {destination}; use a fresh output directory')
            tables.append(table)
            planned.append((destination, table))
        if set(tables[0].Content) & set(tables[1].Content):
            raise ValueError(f'Content overlap in fold {fold}; check source metadata')
    for destination, table in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_excel(destination, index=False)
    print(f'Generated {len(planned)} split tables using user-supplied MOS')


if __name__ == '__main__':
    main()
