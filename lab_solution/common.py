from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import uuid
from pathlib import Path

import numpy as np
SESSION_ID = uuid.uuid4().hex


def serializable(value):
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=serializable), encoding='utf-8')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def environment():
    import torch
    versions = {'python': platform.python_version(), 'platform': platform.platform(),
                'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
    for name in ('torch', 'torchvision', 'timm', 'numpy', 'pandas', 'Pillow', 'matplotlib', 'fvcore', 'openpyxl'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed'
    return versions


def atomic_torch_save(obj, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def code_hashes():
    root = Path(__file__).resolve().parent
    paths = list(root.glob('*.py')) + [root.parent / 'eval.py']
    return {p.name: sha256(p) for p in paths if p.is_file()}
