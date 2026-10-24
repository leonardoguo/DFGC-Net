"""Synthetic host tests: no DA-Net source, dataset or pretrained weights needed."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image

from dfgc.gtfs import make_factor_heads, masked_factor_loss
from dfgc.integration import DFGCExtensions
from dfgc.views import render_six_views, build_mosaic, stable_seed
from dfgc.sjtu_factors import factor_metadata as sjtu_factors
from dfgc.wpc_factors import factor_metadata as wpc_factors
from dfgc.dataset_extension import ContextDataset


class ToyHost(nn.Module):
    """Dimension-only stand-in, intentionally not an implementation of DA-Net."""
    def __init__(self):
        super().__init__()
        self.pc_linear = nn.Linear(3, 1024)
        self.pic_linear = nn.Linear(768, 1024)
        self.pic_backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 768))
        self.fc = nn.Linear(2048, 1)

    def forward(self, pcs, pics):
        batch, patches = pcs.shape[:2]
        geometry = self.pc_linear(pcs.reshape(batch * patches, 3))
        texture = self.pic_linear(self.pic_backbone(pics.flatten(0, 1)))
        return self.fc(torch.cat((geometry, texture), dim=-1).reshape(batch, patches, 2048).mean(1))


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_isolated_head_rng_and_counts(self):
        state = torch.get_rng_state().clone()
        heads = make_factor_heads()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for h in heads for p in h.parameters()), 529410)

    def test_zero_mask(self):
        value = torch.tensor([[0.8], [0.3]], requires_grad=True)
        loss = masked_factor_loss(value, [0., 1.], [0., 0.])
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.equal(value.grad, torch.zeros_like(value)))
        with self.assertRaises(ValueError):
            masked_factor_loss(value, [0., 1.], [0.5, 1.])

    def test_initial_identity_optimizer_gradients_and_reload(self):
        host = ToyHost()
        pcs, pics, mosaic = torch.randn(2, 3, 3), torch.randn(2, 3, 3, 8, 8), torch.randn(2, 3, 224, 224)
        expected = host(pcs, pics).detach()
        extension = DFGCExtensions(host)
        optimizer = torch.optim.AdamW(host.parameters())
        state = torch.get_rng_state().clone()
        with extension.batch(mosaic, 2, 3, epoch=5, step=2) as context:
            prediction = host(pcs, pics)
            auxiliary, terms = context.auxiliary_loss({
                'geometry_severity': torch.tensor([0., 1.]), 'geometry_severity_mask': torch.ones(2),
                'texture_severity': torch.tensor([1., 0.]), 'texture_severity_mask': torch.ones(2)})
            self.assertTrue(torch.equal(expected, prediction.detach()))
            self.assertEqual(terms['geometry_prediction'].shape, (2, 1))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        (prediction.square().mean() + auxiliary).backward()
        for module in (host.gtfs_geometry_head, host.gtfs_texture_head, host.pc_linear,
                       host.pic_linear, host.global_context_adapter[-1]):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None), 0.)
        optimizer.step()
        second = ToyHost()
        DFGCExtensions(second)
        second.load_state_dict(host.state_dict(), strict=True)
        self.assertFalse(host.fc._forward_pre_hooks)
        self.assertFalse(host.pc_linear._forward_hooks)

    def test_exception_cleanup_and_runtime_shape_check(self):
        host = ToyHost()
        extension = DFGCExtensions(host)
        mosaic = torch.zeros(2, 3, 224, 224)
        with self.assertRaises(RuntimeError):
            with extension.batch(mosaic, 2, 3):
                raise RuntimeError('simulated failure')
        self.assertFalse(extension._active)
        self.assertFalse(host.fc._forward_pre_hooks)
        with self.assertRaises(ValueError):
            with extension.batch(mosaic, 2, 4):
                host(torch.zeros(2, 3, 3), torch.zeros(2, 3, 3, 8, 8))
        self.assertFalse(host.pc_linear._forward_hooks)
        with self.assertRaises(ValueError):
            DFGCExtensions(host)

    def test_view_occupancy_and_deterministic_mosaic(self):
        cloud = np.array([[0,0,0,255,255,255], [1,1,1,255,255,255],
                          [0,1,0,255,255,255], [1,0,1,255,255,255]], dtype=float)
        images, masks = render_six_views(cloud)
        self.assertTrue(masks.any())
        self.assertTrue((images == 255).all())
        first = build_mosaic(images, masks, np.random.default_rng(stable_seed('fixed')))
        second = build_mosaic(images, masks, np.random.default_rng(stable_seed('fixed')))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (224, 224, 3))

    def test_factor_maps(self):
        for content in ('redandblack', 'statue', 'ULB Unicorn'):
            for index in range(42):
                labels = sjtu_factors(f'{content}_{index}.ply')
                self.assertIn(labels['geometry_severity_mask'], (0., 1.))
                self.assertIn(labels['texture_severity_mask'], (0., 1.))
        self.assertEqual(wpc_factors('Downsampling', 7, None)['geometry_severity'], 1.)
        self.assertEqual(wpc_factors('Downsampling', 9, None)['texture_severity_mask'], 0.)
        with self.assertRaises(ValueError):
            wpc_factors('Downsampling', 0, None)

    def test_dataset_extension_preserves_local_inputs(self):
        pcs, pics, mos = torch.zeros(2, 4, 6), torch.zeros(2, 3, 224, 224), torch.tensor(.7)
        base = [('statue_0', pcs, pics, mos)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'statue_0'
            directory.mkdir()
            for index in range(6):
                Image.new('RGB', (448, 448), (150, 160, 180)).save(directory / f'view_{index}.png')
            np.savez_compressed(directory / 'occupancy.npz', masks=np.ones((6, 448, 448), dtype=bool))
            (directory / 'manifest.json').write_text('{}', encoding='utf-8')
            wrapper = ContextDataset(base, ['statue_0'], {'statue_0': sjtu_factors('statue_0')},
                                     temporary, training=False)
            metadata, actual_pc, actual_pic, actual_mos = wrapper[0]
            self.assertIs(actual_pc, pcs)
            self.assertIs(actual_pic, pics)
            self.assertIs(actual_mos, mos)
            self.assertEqual(metadata['global_mosaic'].shape, (3, 224, 224))
            self.assertTrue(torch.equal(metadata['global_mosaic'], wrapper[0][0]['global_mosaic']))

    def test_build_splits_from_external_metadata(self):
        import pandas as pd
        root = Path(__file__).resolve().parents[1]
        recipes = json.loads((root / 'configs/splits/sjtu.json').read_text('utf-8'))
        names = recipes['1']['train'] + recipes['1']['vail']
        frame = pd.DataFrame({'Ply_name': names,
                              'Content': [sjtu_factors(name)['content'] for name in names],
                              'MOS': np.linspace(0.1, 9.9, len(names))})
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / 'metadata.csv', Path(temporary) / 'splits'
            frame.to_csv(source, index=False)
            result = subprocess.run([sys.executable, '-B', str(root / 'scripts/build_splits.py'),
                                     '--dataset', 'sjtu', '--metadata', str(source), '--output', str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list(output.rglob('*.xlsx'))), 18)
            self.assertEqual(pd.read_excel(output / 'fold_1/train.xlsx').Ply_name.tolist(), recipes['1']['train'])

    def test_split_recipes_and_release_scope(self):
        root = Path(__file__).resolve().parents[1]
        for dataset, folds, total in [('wpc', 5, 740), ('sjtu', 9, 378)]:
            recipes = json.loads((root / 'configs/splits' / f'{dataset}.json').read_text('utf-8'))
            self.assertEqual(len(recipes), folds)
            held = []
            for recipe in recipes.values():
                train, test = set(recipe['train']), set(recipe['vail'])
                self.assertFalse(train & test)
                self.assertEqual(len(train | test), total)
                held.extend(test)
            self.assertEqual(len(held), total)
            self.assertEqual(len(set(held)), total)
        for relative in ['dfgc/models/_local.py', 'dfgc/models/projection.py', 'dfgc/metrics.py',
                         'scripts/preprocess.py']:
            self.assertFalse((root / relative).exists())
        self.assertFalse(list(root.rglob('*.xlsx')))


if __name__ == '__main__':
    unittest.main()
