"""Reader preparation tests without redistributing upstream implementation."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.io import savemat

from dfgc.preparation import prepare_metadata
from dfgc.runtime import configuration, check_split


class ReaderTests(unittest.TestCase):
    def test_sjtu_mat_to_all_folds_and_safe_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = np.linspace(.1, 9.9, 378).reshape(42, 9)
            savemat(root / 'Final_MOS.mat', {'Final_MOS': values})
            frame = prepare_metadata('sjtu', root, root / 'not_used')
            prepare_metadata('sjtu', root, root / 'not_used')
            self.assertEqual(len(frame), 378)
            self.assertEqual(len(list((root / 'splits').rglob('*.xlsx'))), 18)
            for fold in range(1, 10):
                train, held = check_split(configuration('sjtu', root, fold))
                self.assertEqual((len(train), len(held)), (336, 42))
            row = frame.loc[frame.Ply_name == 'Romanoillamp_0.ply'].iloc[0]
            self.assertEqual(row.MOS, values[0, 4])


if __name__ == '__main__':
    unittest.main()
