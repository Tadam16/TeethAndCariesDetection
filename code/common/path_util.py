"""Consistent paths."""

from pathlib import Path

cache_dir = Path('/cache')
data_dir = Path('/data')
out_dir = Path('/out')
presegment_out_dir = out_dir / 'presegment'
presegment_out_dir.mkdir(exist_ok=True, parents=True)
presegment_predictions_dir = presegment_out_dir / 'predictions'
presegment_predictions_dir.mkdir(exist_ok=True, parents=True)
segment_out_dir = out_dir / 'segment'
segment_out_dir.mkdir(exist_ok=True, parents=True)

preprocess_out_dir = out_dir / 'preprocess'


def rebase(old_base, new_base, path):
    relative_path = path.relative_to(old_base)
    return (new_base / relative_path).resolve()