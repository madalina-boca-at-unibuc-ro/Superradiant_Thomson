"""Per-run output directories outside the source tree."""
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


def create_run_directory(output_root=None):
    root = Path(output_root).expanduser() if output_root is not None else Path.home() / 'output'
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive mkdir guarantees that existing results are never reused.
    while True:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        path = root / f'SRT_{stamp}_{uuid4().hex[:8]}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def _json_default(obj):
    if isinstance(obj, complex):
        return {'real': obj.real, 'imag': obj.imag}
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def write_json(path, data):
    """Replace metadata atomically so readers do not see partial JSON."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False, default=_json_default) + '\n', encoding='utf-8')
    temporary.replace(path)
