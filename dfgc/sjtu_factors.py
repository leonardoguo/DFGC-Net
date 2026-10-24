"""Our SJTU-only GTFS labels; no WPC tables, MOS fitting or random operations.

Source: datasets/SJTU-PCQA/readme.txt (zero-based PLY numbers 0..41).
Targets are relative severity ranks, NOT physical parameter estimates or MOS.
The README's third GGN magnitude (0.02%) is ambiguous; we do not silently
correct it or use physical magnitudes to calculate the training labels.
"""
from pathlib import Path

CONTENTS = ('redandblack', 'ULB Unicorn', 'loot', 'soldier', 'Romanoillamp', 'longdress', 'statue', 'shiva', 'hhi')

SCHEMA = 'sjtu_gtfs_rank6_v1'
SEVERITIES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# (type, geometric factor active, texture factor active).
# OT here is PCL octree pruning, not WPC attribute octree compression.
DISTORTIONS = (
    ('OT', True, False),
    ('CN', False, True),
    ('DS', True, False),
    ('D+C', True, True),
    ('D+G', True, False),
    ('GGN', True, False),
    ('C+G', True, True),
)


def factor_metadata(filename):
    """Decode *original* SJTU sample names. Reject patch names and unknowns.

    Level 1 maps to zero (least severe in this type, NOT pristine).
    Inactive factors have zero placeholders AND zero masks (no supervision).
    All six levels run light -> severe in the official numbering. Downsampling
    retention runs in the reverse direction to removal, but neither is used
    as a filename suffix/continuous physical target here.
    """
    name = str(filename)
    if Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError(f'Expected a SJTU sample name, not a path: {name!r}')
    if name.endswith('.ply'):
        name = name[:-4]
    try:
        content, suffix = name.rsplit('_', 1)
        number = int(suffix)
    except (ValueError, TypeError) as error:
        raise ValueError(f'Invalid SJTU sample name: {name!r}') from error
    if content not in CONTENTS or not 0 <= number < 42 or suffix != str(number):
        raise ValueError(f'Unknown SJTU content/index: {name!r}')
    group, offset = divmod(number, 6)
    kind, geometry, texture = DISTORTIONS[group]
    value = SEVERITIES[offset]
    return dict(name=name, content=content, distortion_index=number,
                distortion_type=kind, severity_level=offset + 1,
                geometry_severity=value if geometry else 0.0,
                geometry_severity_mask=float(geometry),
                texture_severity=value if texture else 0.0,
                texture_severity_mask=float(texture))


def metadata_for_split(frame):
    """Validate a split without changing it or using its MOS to form targets."""
    required = {'Ply_name', 'Content', 'MOS'}
    if not required.issubset(frame.columns):
        raise ValueError(f'SJTU split missing columns: {sorted(required - set(frame.columns))}')
    if frame.Ply_name.duplicated().any():
        raise ValueError('Duplicate SJTU sample in split')
    result = {}
    for row in frame.itertuples(index=False):
        item = factor_metadata(row.Ply_name)
        if row.Content != item['content']:
            raise ValueError(f'SJTU filename/content mismatch: {row.Ply_name}')
        result[item['name']] = item
    return result
