"""Label-free six-face projections and GMS-inspired 7x7 quality mosaics.

Independent implementation, not a copy/reproduction of GMS-3DQA. Uses an
orthographic nearest-surface RGB renderer and explicit occupancy masks, rather
than inferring background from RGB (white surfaces must remain valid).
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
RENDER_VERSION = 1
VIEW_SIZE = 448
GRID = 7
TILE = 32
VIEW_AXES = ((1, 2, 0, 1), (1, 2, 0, -1), (0, 2, 1, 1), (0, 2, 1, -1), (0, 1, 2, 1), (0, 1, 2, -1))

def stable_seed(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:16], 16)

def source_signature(path):
    path = Path(path)
    stat = path.stat()
    return {'source': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'renderer_version': RENDER_VERSION, 'view_size': VIEW_SIZE}

def render_six_views(cloud):
    cloud = np.asarray(cloud)
    if cloud.ndim != 2 or cloud.shape[1] != 6 or len(cloud) == 0:
        raise ValueError('Expected non-empty XYZRGB [N,6] point cloud')
    if not np.isfinite(cloud).all():
        raise ValueError('Non-finite raw point cloud')
    xyz = cloud[:, :3].astype(np.float64)
    (low, high) = (xyz.min(0), xyz.max(0))
    extent = float((high - low).max())
    if extent <= 0:
        raise ValueError('Degenerate raw point cloud')
    xyz = (xyz - (low + high) / 2) / extent
    rgb = np.clip(np.rint(cloud[:, 3:]), 0, 255).astype(np.uint8)
    (images, masks) = ([], [])
    for (horizontal, vertical, depth, direction) in VIEW_AXES:
        x = xyz[:, horizontal] * direction
        y = xyz[:, vertical]
        z = xyz[:, depth] * direction
        u = np.clip(np.rint((x * 0.9 + 0.5) * (VIEW_SIZE - 1)), 0, VIEW_SIZE - 1).astype(np.int64)
        v = np.clip(np.rint((0.5 - y * 0.9) * (VIEW_SIZE - 1)), 0, VIEW_SIZE - 1).astype(np.int64)
        pixel = v * VIEW_SIZE + u
        order = np.lexsort((-z, pixel))
        sorted_pixel = pixel[order]
        keep = np.r_[True, sorted_pixel[1:] != sorted_pixel[:-1]]
        selected = order[keep]
        image = np.full((VIEW_SIZE * VIEW_SIZE, 3), 255, dtype=np.uint8)
        mask = np.zeros(VIEW_SIZE * VIEW_SIZE, dtype=bool)
        image[pixel[selected]] = rgb[selected]
        mask[pixel[selected]] = True
        images.append(image.reshape(VIEW_SIZE, VIEW_SIZE, 3))
        masks.append(mask.reshape(VIEW_SIZE, VIEW_SIZE))
    return (np.stack(images), np.stack(masks))

def cache_complete(directory, source=None):
    directory = Path(directory)
    files = [directory / f'view_{i}.png' for i in range(6)]
    files += [directory / 'occupancy.npz', directory / 'manifest.json']
    if not all((path.is_file() for path in files)):
        return False
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    return source is None or manifest['signature'] == source_signature(source)

def prepare_cloud(source, directory):
    import open3d as o3d
    (source, directory) = (Path(source), Path(directory))
    if cache_complete(directory, source):
        return False
    if (directory / 'manifest.json').exists():
        raise ValueError(f'Stale global-view cache: {directory}; use a new cache directory')
    signature = source_signature(source)
    pcd = o3d.io.read_point_cloud(str(source))
    if not pcd.has_colors():
        raise ValueError(f'Colored XYZRGB input required: {source}')
    cloud = np.concatenate([np.asarray(pcd.points), np.asarray(pcd.colors) * 255.0], axis=1)
    (images, masks) = render_six_views(cloud)
    directory.mkdir(parents=True, exist_ok=True)
    for (index, image) in enumerate(images):
        temporary = directory / f'view_{index}.png.tmp'
        Image.fromarray(image).save(temporary, format='PNG')
        os.replace(str(temporary), str(directory / f'view_{index}.png'))
    temporary = directory / 'occupancy.npz.tmp'
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, masks=masks)
    os.replace(str(temporary), str(directory / 'occupancy.npz'))
    value = {'signature': signature, 'point_count': len(cloud), 'occupancy_fraction': masks.mean(axis=(1, 2)).tolist(), 'view_axes': VIEW_AXES, 'labels_used': False}
    temporary = directory / 'manifest.json.tmp'
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(str(temporary), str(directory / 'manifest.json'))
    return True

def build_mosaic(images, masks, rng):
    (images, masks) = (np.asarray(images), np.asarray(masks))
    if images.shape != (6, VIEW_SIZE, VIEW_SIZE, 3) or masks.shape != (6, VIEW_SIZE, VIEW_SIZE):
        raise ValueError('Unexpected six-view cache shape')
    mosaic = np.empty((224, 224, 3), dtype=np.uint8)
    cell = VIEW_SIZE // GRID
    tiles = []
    for view in range(6):
        candidates = []
        for row in range(GRID):
            for column in range(GRID):
                (top, left) = (row * cell, column * cell)
                region = masks[view, top:top + cell, left:left + cell]
                if region.any():
                    candidates.append((top, left))
        if not candidates:
            raise ValueError(f'Empty rendered view {view}')
        count = 8 if view < 5 else 9
        selected = rng.choice(len(candidates), count, replace=len(candidates) < count)
        for index in selected:
            (top, left) = candidates[int(index)]
            for attempt in range(8):
                (dy, dx) = rng.integers(0, cell - TILE + 1, size=2)
                tile_mask = masks[view, top + dy:top + dy + TILE, left + dx:left + dx + TILE]
                if tile_mask.any():
                    break
            else:
                occupied = np.argwhere(masks[view, top:top + cell, left:left + cell])
                (y, x) = occupied[int(rng.integers(len(occupied)))]
                dy = int(np.clip(y - TILE // 2, 0, cell - TILE))
                dx = int(np.clip(x - TILE // 2, 0, cell - TILE))
            tiles.append(images[view, top + dy:top + dy + TILE, left + dx:left + dx + TILE])
    for (index, tile) in enumerate(tiles):
        (row, column) = divmod(index, GRID)
        mosaic[row * TILE:(row + 1) * TILE, column * TILE:(column + 1) * TILE] = tile
    return mosaic

def load_views(directory):
    directory = Path(directory)
    with np.load(directory / 'occupancy.npz') as payload:
        masks = payload['masks']
    images = []
    for index in range(6):
        with Image.open(directory / f'view_{index}.png') as image:
            images.append(np.array(image.convert('RGB')))
    return (np.stack(images), masks)

