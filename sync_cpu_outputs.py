"""Copy only completed, lightweight CPU-study evidence into this repository.

Checkpoints and data intentionally remain outside Git. A run is synchronized only
after result.json exists, so an in-progress curve can never be presented as a
completed experiment result.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lab_solution.common import sha256, write_json


DEFAULT_STUDY = (Path.home() / '.codex' / 'visualizations' / '2026' / '10' / '03' /
                 '01a10133-0b7a-7873-a4da-69e8e881f33b' / 'cpu_study')
RUN_FILES = ('config.json', 'environment.json', 'metadata.json', 'pretrained.json',
             'profile.json', 'split_checks.json', 'history.csv', 'result.json',
             'val_logits.npz', 'test_complete.json', 'final_calibration.json',
             'final_latency_batch1.json', 'final_latency_batch32.json')


def copy_file(source: Path, target: Path):
    if not source.is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {'path': target.as_posix(), 'sha256': sha256(target), 'bytes': target.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--destination', type=Path, default=Path('artifacts/local_cpu'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    study, destination = args.study.resolve(), args.destination.resolve()
    if not study.is_dir():
        raise FileNotFoundError(study)
    manifest = {'study': str(study), 'completed_runs': [], 'files': [],
                'policy': 'No data, checkpoints, optimizer states, or test-part caches are copied to Git.'}
    for name in ('cpu_training_plan.json', 'cpu_split_checks.json', 'pipeline_checks.json', 'cpu_status.json'):
        source = study/name
        if source.is_file() and not args.dry_run:
            record = copy_file(source, destination/name)
            if record:
                manifest['files'].append(record)
    for result in sorted((study/'runs').glob('*/*/result.json')):
        source_run = result.parent
        relative = source_run.relative_to(study)
        run_record = {'run': relative.as_posix(), 'result_sha256': sha256(result), 'files': []}
        for name in RUN_FILES:
            source = source_run/name
            if source.is_file() and not args.dry_run:
                record = copy_file(source, destination/relative/name)
                if record:
                    run_record['files'].append(record)
        curve = study/'curves'/f'{source_run.parent.name}_{source_run.name}.png'
        if curve.is_file() and not args.dry_run:
            record = copy_file(curve, destination/'curves'/curve.name)
            if record:
                run_record['files'].append(record)
        manifest['completed_runs'].append(run_record)
    if args.dry_run:
        print(f'Would synchronize {len(manifest["completed_runs"])} completed CPU runs from {study}')
        return
    write_json(destination/'sync_manifest.json', manifest)
    print(f'Synchronized {len(manifest["completed_runs"])} completed CPU runs into {destination}')


if __name__ == '__main__':
    main()
