"""Read-only static interface inventory; never imports or copies upstream code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--model-file', default='model/IQA_Model.py')
    parser.add_argument('--class-name', default='MultiModel')
    args = parser.parse_args()
    root = args.upstream.resolve()
    source = (root / args.model_file).resolve()
    if root not in source.parents or not source.is_file():
        raise FileNotFoundError('Supply an existing model file within the separate upstream checkout')
    release = Path(__file__).resolve().parents[1]
    if root == release or release in root.parents:
        raise ValueError('Keep third-party source outside this release directory')
    payload = source.read_bytes()
    tree = ast.parse(payload.decode('utf-8-sig'))
    classes = [item for item in ast.walk(tree) if isinstance(item, ast.ClassDef) and item.name == args.class_name]
    if len(classes) != 1:
        raise ValueError('Expected exactly one named host class')
    attributes = {item.attr for item in ast.walk(classes[0])
                  if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name) and item.value.id == 'self'}
    required = {'pc_linear', 'pic_linear', 'pic_backbone', 'fc'}
    try:
        result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=False)
        commit = result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        commit = None
    print(json.dumps(dict(model_file=str(Path(args.model_file)), git_commit=commit,
                         model_sha256=hashlib.sha256(payload).hexdigest(),
                         host_class=args.class_name, missing_endpoints=sorted(required - attributes),
                         scope='static names only; runtime integration still needs verification'), indent=2))
    if required - attributes:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
