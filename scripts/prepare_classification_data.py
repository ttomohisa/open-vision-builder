#!/usr/bin/env python3
"""Reproduce the fixed 60-photo classification slice, without model computation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import unicodedata
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'tests/fixtures/classification-real-data.json'
MANIFEST_SHA256 = '7d6885edda14c5e08b9a138d5f56ad9ab15b4d9ce3214b1d2befd9ceb5f80805'
ARCHIVE = ROOT / 'artifacts/p1e/data-source/flower_photos.tgz'
SOURCE = {
    'archiveUrl': 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    'sha256': '4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c',
    'bytes': 228813984,
    'licenseFileSha256': '701e91b9e9f46d1f9623fc793a00fd444fe2821f66b01f3a0ff4a73d82f49507',
}
CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
SPLITS = ['train'] * 6 + ['tuning'] * 2 + ['evaluation'] * 4
RANK_PREFIX = 'ovb-classification-flowers-v1'
LICENSE_URL = 'https://creativecommons.org/licenses/by/2.0/'


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def creator_key(name: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', name).casefold().split())


def validate_manifest(manifest: dict) -> None:
    """Reject changed order, splits, identities, source and unsafe member names."""
    try:
        if manifest['schemaVersion'] != 1 or manifest['classes'] != CLASSES or manifest['source'] != SOURCE:
            raise ValueError('Unsupported schema, class order, or source pins')
        if manifest['selection']['rankPrefix'] != RANK_PREFIX or manifest['canonicalTransform']['pillowVersion'] != '11.1.0':
            raise ValueError('Selection or canonical transformation changed')
        images = manifest['images']
        if not isinstance(images, list) or len(images) != 60:
            raise ValueError('Expected 60 declared images')
        unique_fields = ('id', 'creatorKey', 'flickrId', 'originalSha256', 'pngSha256', 'rgbaSha256')
        seen = {field: set() for field in unique_fields}
        previous_rank = ''
        for index, row in enumerate(images):
            label, rank_in_class = divmod(index, 12)
            if type(row['label']) is not int or row['label'] != label or row['split'] != SPLITS[rank_in_class]:
                raise ValueError('Wrong image class, split, or ordering')
            cl = CLASSES[label]
            match = re.fullmatch(r'flower_photos/' + cl + r'/(\d+)_[A-Za-z0-9_]+\.jpg', row['originalPath'])
            if match is None or row['flickrId'] != match[1] or row['id'] != cl + '-' + match[1]:
                raise ValueError('Unsafe or inconsistent image path/id')
            rank = sha256((RANK_PREFIX + '\0' + row['originalPath'].removeprefix('flower_photos/')).encode())
            if row['selectionSha256'] != rank or (rank_in_class and rank <= previous_rank):
                raise ValueError('Image selection rank or ordering changed')
            previous_rank = rank
            if not isinstance(row['creator'], str) or not row['creator'].strip() or row['creatorKey'] != creator_key(row['creator']):
                raise ValueError('Missing or inconsistent creator identity')
            if re.fullmatch(r'https://www\.flickr\.com/photos/[^/?#]+/(?:' + match[1] + r'/)?', row['sourceUrl']) is None or row['licenseUrl'] != LICENSE_URL:
                raise ValueError('Unapproved photo source or license URL')
            if not isinstance(row['attributionLine'], str) or not row['attributionLine'].startswith(row['originalPath'].removeprefix('flower_photos/') + ' CC-BY '):
                raise ValueError('Missing source attribution line')
            for field in ('originalSha256', 'pngSha256', 'rgbaSha256'):
                if not isinstance(row[field], str) or re.fullmatch(r'[0-9a-f]{64}', row[field]) is None:
                    raise ValueError('Invalid SHA-256')
            for field in ('originalBytes', 'pngBytes'):
                if type(row[field]) is not int or not 0 < row[field] <= 20 * 1024 * 1024:
                    raise ValueError('Invalid image byte size')
            if any(type(row[field]) is not int or row[field] <= 0 for field in ('width', 'height')) or row['width'] * row['height'] > 16_000_000:
                raise ValueError('Invalid image dimensions')
            for field in unique_fields:
                if row[field] in seen[field]:
                    raise ValueError(f'Duplicate image identity: {field}')
                seen[field].add(row[field])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed classification data manifest') from error


def reject_symlinks(path: Path) -> None:
    if '..' in path.parts or any(p.is_symlink() for p in (path.absolute(), *path.absolute().parents)):
        raise ValueError('Parent traversal and symlink paths are not permitted')


def verify_file(path: Path, digest: str, size: int) -> None:
    reject_symlinks(path)
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f'File type/size mismatch: {path}')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
            raise ValueError(f'SHA-256 mismatch: {path}')


def canonical_png(raw: bytes, width: int, height: int) -> bytes:
    import PIL
    from PIL import Image, ImageOps
    if PIL.__version__ != '11.1.0':
        raise ValueError('Canonical image preparation requires Pillow 11.1.0')
    with Image.open(io.BytesIO(raw)) as source:
        if source.format != 'JPEG':
            raise ValueError('Declared source image is not JPEG')
        source.load()
        rgba = ImageOps.exif_transpose(source).convert('RGBA')
        if rgba.size != (width, height):
            raise ValueError('Decoded dimensions differ from the fixed manifest')
        clean = Image.frombytes('RGBA', rgba.size, rgba.tobytes())
    stream = io.BytesIO()
    clean.save(stream, 'PNG', compress_level=9, optimize=False)
    return stream.getvalue()


def download_archive(destination: Path) -> None:
    """Download only the pinned public object; never replace existing bytes."""
    reject_symlinks(destination)
    if destination.exists():
        verify_file(destination, SOURCE['sha256'], SOURCE['bytes'])
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.classification-download-', delete=False) as stream:
            temporary = Path(stream.name)
            count, digest = 0, hashlib.sha256()
            with urllib.request.urlopen(SOURCE['archiveUrl'], timeout=60) as response:
                if hasattr(response, 'geturl') and response.geturl() != SOURCE['archiveUrl']:
                    raise ValueError('Unexpected public archive redirect')
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > SOURCE['bytes']:
                        raise ValueError('Archive exceeds pinned length')
                    digest.update(chunk)
                    stream.write(chunk)
            if count != SOURCE['bytes'] or digest.hexdigest() != SOURCE['sha256']:
                raise ValueError('Downloaded archive length/hash mismatch')
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_output(output: Path, manifest: dict, manifest_raw: bytes) -> bool:
    reject_symlinks(output)
    if not output.exists():
        return False
    if not output.is_dir():
        raise ValueError('Output must be a directory')
    names = {p.name for p in output.iterdir()}
    if not names:
        return False
    if names != {'images', 'data-manifest.json'}:
        raise ValueError('Output contains unrelated or incomplete files')
    images = output / 'images'
    reject_symlinks(images)
    if not images.is_dir() or {p.name for p in images.iterdir()} != {row['id'] + '.png' for row in manifest['images']}:
        raise ValueError('Output image set differs from the fixed manifest')
    verify_file(output / 'data-manifest.json', sha256(manifest_raw), len(manifest_raw))
    for row in manifest['images']:
        verify_file(images / (row['id'] + '.png'), row['pngSha256'], row['pngBytes'])
    return True


def prepare(manifest_path: Path, output_dir: Path, archive_path: Path | None = None) -> dict:
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    reject_symlinks(manifest_path)
    raw = manifest_path.read_bytes()
    if sha256(raw) != MANIFEST_SHA256:
        raise ValueError('Manifest bytes differ from the preregistered fixture')
    manifest = json.loads(raw)
    validate_manifest(manifest)
    if archive_path is not None:
        archive_path = Path(archive_path)
        reject_symlinks(archive_path)
    if verify_output(output_dir, manifest, raw):
        return manifest
    if archive_path is None:
        archive_path = ARCHIVE
        download_archive(archive_path)
    verify_file(archive_path, SOURCE['sha256'], SOURCE['bytes'])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.classification-data-', dir=output_dir.parent))
    try:
        (stage / 'images').mkdir()
        declared = {row['originalPath']: row for row in manifest['images']}
        found = set()
        license_text = None
        with tarfile.open(archive_path, 'r|gz') as archive:
            for member in archive:
                is_license = member.name == 'flower_photos/LICENSE.txt'
                if member.name not in declared and not is_license:
                    continue
                if member.name in found or not member.isfile():
                    raise ValueError('Repeated or non-regular declared archive member')
                found.add(member.name)
                expected_size = 2 * 1024 * 1024 if is_license else declared[member.name]['originalBytes']
                if member.size > expected_size or (not is_license and member.size != expected_size):
                    raise ValueError('Declared archive member has wrong size')
                member_raw = archive.extractfile(member).read()
                if is_license:
                    if sha256(member_raw) != SOURCE['licenseFileSha256']:
                        raise ValueError('Archived image license hash mismatch')
                    license_text = member_raw.decode('utf-8')
                    continue
                row = declared[member.name]
                if sha256(member_raw) != row['originalSha256']:
                    raise ValueError('Original JPEG hash mismatch')
                png = canonical_png(member_raw, row['width'], row['height'])
                if len(png) != row['pngBytes'] or sha256(png) != row['pngSha256']:
                    raise ValueError('Canonical PNG differs from the pinned bytes')
                (stage / 'images' / (row['id'] + '.png')).write_bytes(png)
        if found != set(declared) | {'flower_photos/LICENSE.txt'}:
            raise ValueError('Declared images or license missing from archive')
        license_lines = set(license_text.splitlines())
        if any(row['attributionLine'] not in license_lines for row in manifest['images']):
            raise ValueError('Image attribution is absent from the archived license')
        (stage / 'data-manifest.json').write_bytes(raw)
        reject_symlinks(output_dir)
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise ValueError('Output changed during preparation')
        os.rename(stage, output_dir)
    except (tarfile.TarError, OSError) as error:
        raise ValueError(f'Image/archive preparation failed: {error}') from error
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/p1e/data')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    manifest = prepare(args.manifest, args.output, args.archive)
    print(json.dumps({'prepared': True, 'images': len(manifest['images']), 'splitCounts': dict(Counter(row['split'] for row in manifest['images'])), 'manifestSha256': MANIFEST_SHA256, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
