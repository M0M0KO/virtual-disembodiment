"""Atomic, versioned artifacts for the real-data retrospective pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def require_same_or_new(path, value):
    """A freeze cannot be replaced with a different split/configuration."""
    path = Path(path)
    if path.exists():
        old = read_json(path)
        if object_hash(old) != object_hash(value):
            raise ValueError('Existing freeze differs; keep it and choose a new run: ' + str(path))
    else:
        atomic_json(path, value)


def code_inventory(project):
    project = Path(project)
    return {str(p.relative_to(project)): file_hash(p)
            for directory in ('src', 'scripts', 'configs')
            for p in sorted((project/directory).rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.json') and '__pycache__' not in p.parts}
