"""Guard the fixed experiment data boundary, without network or inference."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/prepare_detector_impact_data.py'
MANIFEST = ROOT / 'tests/fixtures/detector-impact-data.json'


class ImpactDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SCRIPT.exists():
            spec = importlib.util.spec_from_file_location('impact_data', SCRIPT)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        else:
            cls.module = None

    def setUp(self):
        self.assertIsNotNone(self.module, 'Fixed data preparation helper is not implemented')

    def fixture(self):
        return json.loads(MANIFEST.read_text())

    def test_pixel_xywh_becomes_normalized_cxcywh_with_sparse_category_mapping(self):
        ann = [{'id': 7, 'category_id': 23, 'bbox': [10, 20, 40, 30], 'iscrowd': 0}]
        got = self.module.normalize_annotations(ann, 100, 100, [1, 17, 23])
        self.assertEqual(got, {'labels': [2], 'boxes': [0.3, 0.35, 0.4, 0.3]})

    def test_invalid_boxes_and_unknown_categories_are_rejected(self):
        for box in ([0, 0, 0, 2], [-1, 0, 3, 2], [99, 0, 2, 2], [0, 0, float('nan'), 2]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                self.module.normalize_annotations([{'category_id': 1, 'bbox': box, 'iscrowd': 0}], 100, 100, [1])
        with self.assertRaises(ValueError):
            self.module.normalize_annotations([{'category_id': 17, 'bbox': [0, 0, 2, 2], 'iscrowd': 0}], 100, 100, [1])

    def test_committed_data_has_complete_annotations_and_disjoint_fixed_splits(self):
        manifest = self.fixture()
        self.module.validate_manifest(manifest)
        self.assertEqual([x['split'] for x in manifest['images']], ['train'] * 4 + ['tuning'] * 2 + ['evaluation'] * 4)
        self.assertEqual(sum(len(x['targets']['labels']) for x in manifest['images']), sum(len(x['annotations']) for x in manifest['images']))

    def test_duplicate_images_wrong_splits_and_changed_targets_are_rejected(self):
        for change in ('duplicate', 'split', 'target'):
            manifest = self.fixture()
            if change == 'duplicate':
                manifest['images'][-1]['id'] = manifest['images'][0]['id']
            elif change == 'split':
                manifest['images'][-1]['split'] = 'train'
            else:
                manifest['images'][0]['targets']['boxes'][0] = 1.1
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.module.validate_manifest(manifest)

    def test_arbitrary_urls_paths_and_invalid_hashes_are_rejected(self):
        for change in ('url', 'path', 'hash'):
            manifest = self.fixture()
            first = manifest['images'][0]
            if change == 'url':
                first['url'] = 'https://example.org/untrusted.jpg'
            elif change == 'path':
                first['input']['file'] = '../escaped.f32'
            else:
                first['sha256'] = 'not-a-hash'
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.module.validate_manifest(manifest)

    def test_split_and_annotation_digest_mismatches_are_rejected(self):
        for field in ('splitSha256', 'annotationSubsetSha256'):
            manifest = self.fixture()
            manifest[field] = '0' * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.module.validate_manifest(manifest)

    def test_data_set_hash_failure_does_not_publish_partial_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'data'
            source = destination / 'source'
            source.mkdir(parents=True)
            marker = source / 'existing-source.txt'
            marker.write_text('keep this source')
            with patch.object(self.module.urllib.request, 'urlopen', return_value=io.BytesIO(b'bad image')):
                with self.assertRaises(ValueError):
                    self.module.prepare_data(self.fixture(), destination, '0' * 64)
            self.assertEqual({p.name for p in destination.iterdir()}, {'source'})
            self.assertEqual(marker.read_text(), 'keep this source')
            self.assertEqual({p.name for p in Path(tmp).iterdir()}, {'data'})

    def test_preparation_rejects_a_symlink_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / 'outside'
            outside.mkdir()
            destination = Path(tmp) / 'data'
            destination.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.module.prepare_data(self.fixture(), destination, '0' * 64)
            self.assertEqual(list(outside.iterdir()), [])

    def test_bad_download_hash_leaves_destination_unchanged(self):
        payload = b'wrong bytes'
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'image.jpg'
            with patch.object(self.module.urllib.request, 'urlopen', return_value=io.BytesIO(payload)):
                with self.assertRaises(ValueError):
                    self.module.download_verified('https://s3.amazonaws.com/images.cocodataset.org/val2017/000000000285.jpg', target, '0' * 64, len(payload))
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_existing_unrelated_file_and_symlink_are_not_overwritten(self):
        payload = b'original'
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / 'original.jpg'
            original.write_bytes(payload)
            link = Path(tmp) / 'link.jpg'
            link.symlink_to(original)
            with self.assertRaises(ValueError):
                self.module.download_verified('https://s3.amazonaws.com/images.cocodataset.org/val2017/000000000285.jpg', original, '0' * 64, len(payload))
            with self.assertRaises(ValueError):
                self.module.download_verified('https://s3.amazonaws.com/images.cocodataset.org/val2017/000000000285.jpg', link, expected, len(payload))
            self.assertEqual(original.read_bytes(), payload)

    def test_rgb_nchw_float32_div255_and_no_exif_rotation(self):
        from PIL import Image
        import numpy as np
        stream = io.BytesIO()
        Image.new('RGB', (2, 1), (255, 128, 0)).save(stream, format='PNG')
        raw = self.module.preprocess_image(stream.getvalue(), 2, 1)
        values = np.frombuffer(raw, dtype='<f4').reshape(1, 3, 640, 640)
        self.assertEqual(len(raw), 4915200)
        self.assertEqual(float(values[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(values[0, 1, 0, 0]), float(np.float32(128) / np.float32(255)))
        self.assertEqual(float(values[0, 2, 0, 0]), 0.0)


if __name__ == '__main__':
    unittest.main()
