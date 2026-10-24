"""The rectangle diagnostic must preserve failures and reject invalid evidence."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import reproduce_detector_rectangle as rectangle


class RectangleReproduction(unittest.TestCase):
    def test_known_input_matches_the_recorded_bytes(self):
        image = rectangle.rectangle_input()
        self.assertEqual(image.shape, (1, 3, 640, 640))
        self.assertEqual(image.dtype, np.float32)
        self.assertEqual(hashlib.sha256(image.tobytes()).hexdigest(),
                         'dad00b5c87bfc67d5b12f1212f2c88aeab24c6944998b2200f1e28ab9c0b42eb')

    def test_numeric_mismatch_is_a_complete_failed_comparison(self):
        reference = np.zeros((1, 300, 4), dtype=np.float32)
        actual = reference.copy(); actual[0, 0, 0] = .001
        result = rectangle.compare_tensor(actual, reference, 'boxes')
        self.assertFalse(result['passedAtOriginalBound'])
        self.assertEqual(result['failedElements'], 1)
        self.assertEqual(result['elements'], 1200)
        self.assertEqual(result['atol'], 2e-5)
        self.assertEqual(result['rtol'], 1e-4)

    def test_rejects_wrong_shape_dtype_and_nonfinite_values(self):
        reference = np.zeros((1, 300, 4), dtype=np.float32)
        bad = [np.zeros((300, 4), np.float32), reference.astype(np.float64),
               np.full_like(reference, np.nan), np.full_like(reference, np.inf)]
        for value in bad:
            with self.subTest(dtype=value.dtype, shape=value.shape), self.assertRaises(ValueError):
                rectangle.compare_tensor(value, reference, 'boxes')
        with self.assertRaises(ValueError):
            rectangle.compare_tensor(reference[:, :1], reference[:, :1], 'boxes')

    def test_failed_run_invalidates_old_completed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text(json.dumps({'kind': rectangle.REPORT_KIND, 'status': 'completed', 'sourceParityPassed': True}))
            def fail(*args):
                self.assertEqual(json.loads(path.read_text())['status'], 'running')
                raise ValueError('broken model input')
            with patch.object(rectangle, 'diagnose', side_effect=fail):
                self.assertEqual(rectangle.execute(path, Path('assets'), Path('source')), 1)
            result = json.loads(path.read_text())
            self.assertEqual(result['status'], 'error')
            self.assertFalse(result['qualifiesDetector'])
            self.assertNotIn('sourceParityPassed', result)

    def test_interrupted_run_never_keeps_a_completed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text(json.dumps({'kind': rectangle.REPORT_KIND, 'status': 'completed', 'sourceParityPassed': True}))
            with patch.object(rectangle, 'diagnose', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                rectangle.execute(path, Path('assets'), Path('source'))
            self.assertEqual(json.loads(path.read_text())['status'], 'running')

    def test_preserves_unowned_file_and_rejects_symlink_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'; path.write_text('keep')
            link = Path(directory) / 'link.json'; link.symlink_to(path)
            for output in (path, link):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    rectangle.execute(output, Path('assets'), Path('source'))
            self.assertEqual(path.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
