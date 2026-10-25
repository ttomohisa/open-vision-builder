#!/usr/bin/env python3
"""Reproduce the preregistered COCO slice; no selection, training, or inference.

Use the detector venv (Pillow 11.1.0 / NumPy 2.2.3). Original JPEGs and
canonical tensors stay under ignored artifacts/p1d/data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'tests/fixtures/detector-impact-data.json'
ARCHIVE_URL = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'
ARCHIVE_SHA256 = '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'
ANNOTATION_SHA256 = 'e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f'
IMAGE_BASE = 'https://s3.amazonaws.com/images.cocodataset.org/val2017/'
INPUT_BYTES = 4915200
ALIASES = {'motorcycle': 'motorbike', 'airplane': 'aeroplane', 'couch': 'sofa', 'potted plant': 'pottedplant', 'dining table': 'diningtable', 'tv': 'tvmonitor'}
COCO_IDS = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
PREPROCESSING = {'library': 'Pillow', 'version': '11.1.0', 'numpyVersion': '2.2.3', 'color': 'RGB', 'resize': [640, 640], 'resampling': 'BILINEAR', 'geometry': 'full-image stretch; no padding or cropping', 'exifTranspose': False, 'layout': 'NCHW', 'dtype': 'float32-le', 'rescale': 'float32 RGB / float32(255)', 'shape': [1, 3, 640, 640]}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_annotations(annotations: list, width: int, height: int, category_ids: list) -> dict:
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions')
    mapping = {category_id: index for index, category_id in enumerate(category_ids)}
    labels, boxes = [], []
    for annotation in annotations:
        if annotation.get('iscrowd') != 0 or annotation['category_id'] not in mapping:
            raise ValueError('Crowd or unknown category')
        bbox = annotation['bbox']
        if not isinstance(bbox, list) or len(bbox) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox):
            raise ValueError('Invalid bounding box')
        x, y, w, h = bbox
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width + 1e-5 or y + h > height + 1e-5:
            raise ValueError('Bounding box outside image or nonpositive')
        labels.append(mapping[annotation['category_id']])
        boxes.extend([(x + w / 2) / width, (y + h / 2) / height, w / width, h / height])
    return {'labels': labels, 'boxes': boxes}


def validate_manifest(manifest: dict) -> None:
    if manifest.get('schemaVersion') != 1 or manifest.get('preprocessing') != PREPROCESSING:
        raise ValueError('Unsupported schema or preprocessing')
    sources = manifest['sources']
    if sources['annotationsArchive']['sha256'] != ARCHIVE_SHA256 or sources['instances']['sha256'] != ANNOTATION_SHA256:
        raise ValueError('Annotation source hash changed')
    categories = manifest['categories']
    if [c['id'] for c in categories] != COCO_IDS:
        raise ValueError('COCO80 category ordering changed')
    if any(c['modelLabel'] != ALIASES.get(c['name'], c['name']) for c in categories):
        raise ValueError('COCO category/model label mismatch')
    images = manifest['images']
    if len(images) != 10 or [im['split'] for im in images] != ['train'] * 4 + ['tuning'] * 2 + ['evaluation'] * 4:
        raise ValueError('Expected exactly the fixed 4/2/4 split')
    ids = [im['id'] for im in images]
    if any(not isinstance(i, str) or re.fullmatch(r'\d{12}', i) is None for i in ids) or ids != sorted(set(ids)):
        raise ValueError('Images must have distinct ascending COCO ids')
    if len({im['sha256'] for im in images}) != len(images):
        raise ValueError('Duplicate image bytes across split')
    if len({im['input']['sha256'] for im in images}) != len(images):
        raise ValueError('Duplicate input bytes across split')
    annotation_ids = set()
    for im in images:
        if im['url'] != IMAGE_BASE + im['id'] + '.jpg' or im['input']['file'] != 'inputs/' + im['id'] + '.f32':
            raise ValueError('Unapproved URL or input path')
        for item in (im, im['input']):
            if not isinstance(item.get('sha256'), str) or re.fullmatch(r'[0-9a-f]{64}', item['sha256']) is None:
                raise ValueError('Invalid SHA-256')
        if not isinstance(im['bytes'], int) or not 0 < im['bytes'] <= 20 * 1024 * 1024:
            raise ValueError('Invalid image byte count')
        if im['input']['bytes'] != INPUT_BYTES or im['input']['shape'] != [1, 3, 640, 640]:
            raise ValueError('Invalid input tensor shape or size')
        if im['license'] != {'id': 4, 'name': 'CC BY 2.0', 'url': 'https://creativecommons.org/licenses/by/2.0/'}:
            raise ValueError('Image license is not CC BY 2.0')
        attr = im['attribution']
        if not attr.get('creator') or not attr.get('title') or not attr['sourcePage'].startswith('https://www.flickr.com/photos/') or attr.get('license') != 'CC BY 2.0':
            raise ValueError('Missing source attribution')
        annotations = im['annotations']
        if not 1 <= len(annotations) <= 12:
            raise ValueError('Instance count outside preregistered range')
        for annotation in annotations:
            if annotation['id'] in annotation_ids or annotation['image_id'] != int(im['id']):
                raise ValueError('Duplicate or wrong-image annotation')
            annotation_ids.add(annotation['id'])
        expected = normalize_annotations(annotations, im['width'], im['height'], COCO_IDS)
        if im['targets'] != expected:
            raise ValueError('Targets do not match all source annotations')
    for field, values in (
        ('splitSha256', [{'id': im['id'], 'split': im['split']} for im in images]),
        ('annotationSubsetSha256', [{'id': im['id'], 'annotations': im['annotations']} for im in images]),
    ):
        encoded = json.dumps(values, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
        if manifest.get(field) != sha256(encoded):
            raise ValueError('Split or annotation subset digest mismatch')


def reject_symlinks(path: Path) -> None:
    absolute = path.absolute()
    if any(p.is_symlink() for p in (absolute, *absolute.parents)):
        raise ValueError('Symlink paths are not permitted')


def verify_file(path: Path, expected_sha256: str, expected_bytes: int) -> None:
    reject_symlinks(path)
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError(f'File size/type mismatch: {path}')
    if sha256(path.read_bytes()) != expected_sha256:
        raise ValueError(f'SHA-256 mismatch: {path}')


def download_verified(url: str, destination: Path, expected_sha256: str, expected_bytes: int) -> None:
    if url != ARCHIVE_URL and re.fullmatch(re.escape(IMAGE_BASE) + r'\d{12}\.jpg', url) is None:
        raise ValueError('Download URL is not an approved COCO object')
    reject_symlinks(destination)
    if destination.exists():
        verify_file(destination, expected_sha256, expected_bytes)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.download-', delete=False) as stream:
            temporary = Path(stream.name)
            digest, count = hashlib.sha256(), 0
            with urllib.request.urlopen(url, timeout=30) as response:
                final_url = response.geturl() if hasattr(response, 'geturl') else url
                if final_url != url:
                    raise ValueError('Unexpected download redirect')
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > expected_bytes:
                        raise ValueError('Download exceeds pinned length')
                    digest.update(chunk)
                    stream.write(chunk)
            if count != expected_bytes or digest.hexdigest() != expected_sha256:
                raise ValueError('Downloaded bytes/hash do not match the manifest')
        # link is atomic and refuses to overwrite a file created concurrently.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def preprocess_image(raw: bytes, width: int, height: int) -> bytes:
    import PIL
    from PIL import Image
    import numpy as np
    if PIL.__version__ != '11.1.0' or np.__version__ != '2.2.3':
        raise ValueError('Preprocessing requires Pillow 11.1.0 and NumPy 2.2.3')
    with Image.open(io.BytesIO(raw)) as image:
        if image.size != (width, height):
            raise ValueError('Decoded dimensions differ from annotation metadata')
        image = image.convert('RGB').resize((640, 640), Image.Resampling.BILINEAR)
        data = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / np.float32(255)
        return np.ascontiguousarray(data, dtype='<f4').tobytes()


def verify_model_labels(manifest: dict, model_path: Path) -> None:
    model = json.loads(model_path.read_text())
    if model['labels'] != [c['modelLabel'] for c in manifest['categories']]:
        raise ValueError('Prepared model label order differs from the COCO80 mapping')


def prepare_data(manifest: dict, destination: Path, manifest_hash: str) -> None:
    validate_manifest(manifest)
    reject_symlinks(destination)
    if destination.exists():
        if not destination.is_dir():
            raise ValueError('Destination must be a directory')
        names = {p.name for p in destination.iterdir()}
        if 'prepared.json' in names:
            marker_path = destination / 'prepared.json'
            reject_symlinks(marker_path)
            marker = json.loads(marker_path.read_text())
            if names != {'source', 'images', 'inputs', 'prepared.json'} and names != {'images', 'inputs', 'prepared.json'}:
                raise ValueError('Destination contains unrelated files')
            if marker.get('manifestSha256') != manifest_hash:
                raise ValueError('Destination belongs to a different manifest')
            for im in manifest['images']:
                verify_file(destination / 'images' / (im['id'] + '.jpg'), im['sha256'], im['bytes'])
                verify_file(destination / im['input']['file'], im['input']['sha256'], INPUT_BYTES)
            return
        if names - {'source'}:
            raise ValueError('Refusing to overwrite an unrelated or incomplete destination')
        if 'source' in names:
            reject_symlinks(destination / 'source')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.impact-data-', dir=destination.parent))
    try:
        (stage / 'images').mkdir()
        (stage / 'inputs').mkdir()
        for im in manifest['images']:
            image_path = stage / 'images' / (im['id'] + '.jpg')
            download_verified(im['url'], image_path, im['sha256'], im['bytes'])
            tensor = preprocess_image(image_path.read_bytes(), im['width'], im['height'])
            if len(tensor) != INPUT_BYTES or sha256(tensor) != im['input']['sha256']:
                raise ValueError('Preprocessed input differs from pinned bytes')
            (stage / im['input']['file']).write_bytes(tensor)
        (stage / 'prepared.json').write_text(json.dumps({'schemaVersion': 1, 'manifestSha256': manifest_hash, 'images': 10}, indent=2) + '\n')
        reject_symlinks(destination)
        destination.mkdir(exist_ok=True)
        if {p.name for p in destination.iterdir()} - {'source'}:
            raise ValueError('Destination changed during preparation')
        # Publish only after every source and tensor hash passes; marker is last.
        for name in ('images', 'inputs', 'prepared.json'):
            os.rename(stage / name, destination / name)
    finally:
        shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/p1d/data')
    parser.add_argument('--model', type=Path, default=ROOT / 'artifacts/p1b/detector-assets/unqualified/model.json')
    args = parser.parse_args()
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    validate_manifest(manifest)
    verify_model_labels(manifest, args.model)
    prepare_data(manifest, args.output, sha256(raw))
    print(json.dumps({'prepared': True, 'images': 10, 'splitCounts': dict(Counter(im['split'] for im in manifest['images'])), 'manifestSha256': sha256(raw), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
