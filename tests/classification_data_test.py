"""The fixed classification data boundary: no network, model, or inference."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/prepare_classification_data.py'
MANIFEST = ROOT / 'tests/fixtures/classification-real-data.json'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class ClassificationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SCRIPT.exists():
            spec = importlib.util.spec_from_file_location('classification_data', SCRIPT)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        else:
            cls.module = None

    def setUp(self):
        self.assertIsNotNone(self.module, 'Classification data preparation is not implemented')

    def fixture(self):
        return json.loads(MANIFEST.read_text())

    def test_fixed_manifest_has_disjoint_ordered_splits_and_attribution(self):
        data = self.fixture()
        self.module.validate_manifest(data)
        self.assertEqual([row['split'] for row in data['images']], (['train'] * 6 + ['tuning'] * 2 + ['evaluation'] * 4) * 5)
        self.assertEqual(len({row['creatorKey'] for row in data['images']}), 60)
        self.assertTrue(all(row['attributionLine'] for row in data['images']))

    def test_duplicate_identity_wrong_label_and_swapped_split_are_rejected(self):
        for field in ('id', 'creatorKey', 'flickrId', 'originalSha256', 'pngSha256', 'rgbaSha256'):
            data = self.fixture()
            data['images'][1][field] = data['images'][0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.module.validate_manifest(data)
        for mutation in ('label', 'order', 'split', 'class-order', 'creator'):
            data = self.fixture()
            if mutation == 'label':
                data['images'][0]['label'] = 1
            elif mutation == 'order':
                data['images'][0], data['images'][1] = data['images'][1], data['images'][0]
            elif mutation == 'split':
                data['images'][0]['split'], data['images'][6]['split'] = 'tuning', 'train'
            elif mutation == 'class-order':
                data['classes'].reverse()
            else:
                data['images'][0]['creator'] = 'Someone else'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.module.validate_manifest(data)

    def test_unapproved_archive_paths_urls_hashes_and_dimensions_are_rejected(self):
        cases = [('originalPath', '../escaped.jpg'), ('originalPath', '/tmp/escaped.jpg'),
                 ('id', '../escaped'), ('sourceUrl', 'https://example.org/photo'),
                 ('licenseUrl', 'https://example.org/license'), ('pngSha256', 'not-sha256'),
                 ('width', 0), ('height', True), ('pngBytes', 30 * 1024 * 1024)]
        for field, value in cases:
            data = self.fixture()
            data['images'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.module.validate_manifest(data)
        data = self.fixture()
        data['source']['archiveUrl'] = 'https://example.org/untrusted.tgz'
        with self.assertRaises(ValueError):
            self.module.validate_manifest(data)

    def test_canonical_png_applies_exif_and_strips_metadata_without_resizing(self):
        jpeg = io.BytesIO()
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (3, 2), (255, 0, 0)).save(jpeg, 'JPEG', quality=95, exif=exif)
        raw = self.module.canonical_png(jpeg.getvalue(), 2, 3)
        with Image.open(io.BytesIO(raw)) as image:
            self.assertEqual((image.format, image.mode, image.size), ('PNG', 'RGBA', (2, 3)))
            self.assertEqual(image.getpixel((0, 0)), (254, 0, 0, 255))
            self.assertEqual(image.getexif().get(274), None)
        with self.assertRaises(ValueError):
            self.module.canonical_png(jpeg.getvalue(), 3, 2)

    @contextmanager
    def tiny_source(self, directory, mutation=None):
        """Replace only public-source pins; exercise real archive/filesystem processing."""
        data = self.fixture()
        archive = directory / 'source.tgz'
        license_bytes = (data['attribution']['archiveLicenseHeader'] + '\n\n' + '\n'.join(row['attributionLine'] for row in data['images']) + '\n').encode()
        with tarfile.open(archive, 'w:gz') as tar:
            item = tarfile.TarInfo('flower_photos/LICENSE.txt')
            item.size = len(license_bytes)
            tar.addfile(item, io.BytesIO(license_bytes))
            for index, row in enumerate(data['images']):
                jpeg = io.BytesIO()
                Image.new('RGB', (3, 2), (index * 4, index * 3, index * 2)).save(jpeg, 'JPEG', quality=95)
                raw = jpeg.getvalue()
                with Image.open(io.BytesIO(raw)) as decoded:
                    rgba = decoded.convert('RGBA')
                    png = io.BytesIO()
                    rgba.save(png, 'PNG', compress_level=9, optimize=False)
                row.update(originalSha256=digest(raw), originalBytes=len(raw), pngSha256=digest(png.getvalue()),
                           pngBytes=len(png.getvalue()), width=3, height=2, rgbaSha256=digest(rgba.tobytes()))
                item = tarfile.TarInfo(row['originalPath'])
                item.size = len(raw)
                if mutation == 'selected-link' and index == 0:
                    item.type = tarfile.SYMTYPE
                    item.linkname = '/tmp/must-not-read'
                    item.size = 0
                    tar.addfile(item)
                else:
                    tar.addfile(item, io.BytesIO(raw))
            # An undeclared unsafe member must never be extracted.
            item = tarfile.TarInfo('../escaped.txt')
            item.size = 6
            tar.addfile(item, io.BytesIO(b'escape'))
        source = dict(data['source'], sha256=digest(archive.read_bytes()), bytes=archive.stat().st_size,
                      licenseFileSha256=digest(license_bytes))
        data['source'] = source
        if mutation == 'jpeg-hash':
            data['images'][0]['originalSha256'] = '0' * 64
        elif mutation == 'png-hash':
            data['images'][0]['pngSha256'] = '0' * 64
        elif mutation == 'license-hash':
            source['licenseFileSha256'] = '0' * 64
        manifest = directory / 'fixture.json'
        manifest.write_text(json.dumps(data, ensure_ascii=False) + '\n')
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.module, 'SOURCE', source))
            stack.enter_context(patch.object(self.module, 'MANIFEST_SHA256', digest(manifest.read_bytes())))
            yield manifest, archive, data

    def test_prepare_publishes_only_declared_pngs_and_verifies_idempotent_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.tiny_source(root) as (manifest, archive, data):
                output = root / 'data'
                self.module.prepare(manifest, output, archive)
                self.assertEqual({p.name for p in output.iterdir()}, {'images', 'data-manifest.json'})
                self.assertEqual({p.name for p in (output / 'images').iterdir()}, {r['id'] + '.png' for r in data['images']})
                self.assertEqual((output / 'data-manifest.json').read_bytes(), manifest.read_bytes())
                self.assertFalse((root / 'escaped.txt').exists())
                first = output / 'images' / (data['images'][0]['id'] + '.png')
                before = first.stat().st_mtime_ns
                self.module.prepare(manifest, output, archive)
                self.assertEqual(first.stat().st_mtime_ns, before)
                with Image.open(first) as im:
                    self.assertEqual(im.getpixel((0, 0)), (0, 0, 0, 255))

    def test_archive_license_jpeg_png_or_member_corruption_publishes_nothing(self):
        for mutation in ('archive-hash', 'license-hash', 'jpeg-hash', 'png-hash', 'selected-link'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.tiny_source(root, mutation) as (manifest, archive, _):
                    if mutation == 'archive-hash':
                        archive.write_bytes(archive.read_bytes() + b'tampered')
                    output = root / 'data'
                    with self.assertRaises(ValueError):
                        self.module.prepare(manifest, output, archive)
                    self.assertFalse(output.exists())
                    self.assertFalse(any(p.name.startswith('.classification-') for p in root.iterdir()))

    def test_unrelated_output_and_symlink_paths_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.tiny_source(root) as (manifest, archive, _):
                output = root / 'data'
                output.mkdir()
                sentinel = output / 'user-file.txt'
                sentinel.write_text('keep')
                with self.assertRaises(ValueError):
                    self.module.prepare(manifest, output, archive)
                self.assertEqual(sentinel.read_text(), 'keep')
                linked = root / 'linked'
                linked.symlink_to(output, target_is_directory=True)
                with self.assertRaises(ValueError):
                    self.module.prepare(manifest, linked / 'child', archive)
                self.assertFalse((output / 'child').exists())
                for kind, target in [('manifest', manifest), ('archive', archive)]:
                    link = root / (kind + '-link')
                    link.symlink_to(target)
                    with self.subTest(kind=kind), self.assertRaises(ValueError):
                        self.module.prepare(link if kind == 'manifest' else manifest, root / 'unused', link if kind == 'archive' else archive)

    def test_existing_outputs_reject_tampered_png_extra_files_and_symlinks(self):
        for mutation in ('png', 'extra', 'symlink', 'manifest'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.tiny_source(root) as (manifest, archive, data):
                    output = root / 'data'
                    self.module.prepare(manifest, output, archive)
                    first = output / 'images' / (data['images'][0]['id'] + '.png')
                    if mutation == 'png':
                        first.write_bytes(first.read_bytes() + b'tampered')
                    elif mutation == 'extra':
                        (output / 'images' / 'extra.png').write_text('keep')
                    elif mutation == 'symlink':
                        first.unlink()
                        first.symlink_to(archive)
                    else:
                        (output / 'data-manifest.json').write_text('{}')
                    with self.assertRaises(ValueError):
                        self.module.prepare(manifest, output, archive)

    def test_changed_manifest_bytes_are_rejected_before_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'changed.json'
            manifest.write_bytes(MANIFEST.read_bytes() + b' ')
            with self.assertRaises(ValueError):
                self.module.prepare(manifest, root / 'data', root / 'missing.tgz')
            self.assertFalse((root / 'data').exists())


if __name__ == '__main__':
    unittest.main()
