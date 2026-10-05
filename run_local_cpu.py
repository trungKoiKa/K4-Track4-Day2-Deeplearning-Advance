"""Continue the six final training runs locally, without opening test results.

The historical Colab runs and lab_solution sources are left byte-for-byte intact.
An immutable manifest records this driver, the training implementation, data and
CPU environment. Inference selection and final test require a separate val lock.
"""
from __future__ import annotations

import argparse
import gc
import os
import platform
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

from lab_solution import train
from lab_solution.common import code_hashes, environment, read_json, sha256, write_json
from lab_solution.dataset import check_split, load_split
from lab_solution.sanity import pipeline_checks


class ProgressLoader:
    """Observe progress without changing sampling, augmentation or optimization."""

    def __init__(self, loader, report):
        self.loader, self.report = loader, report

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for index, item in enumerate(self.loader):
            if index % 16 == 0:
                self.report(index, len(self.loader))
            yield item
        self.report(len(self.loader), len(self.loader))


def configs(root, data, workers):
    result = []
    for exp in ('T00', 'F01'):
        for seed in (0, 1, 2):
            result.append(train.Config(
                exp_id=exp, seed=seed, backbone='convnext_tiny',
                pretrained_tag='convnext_tiny.fb_in1k', epochs=10,
                batch_size=16, grad_accum=4, amp=False, device='cpu',
                num_workers=workers, loss='ce' if exp == 'T00' else 'ls',
                label_smoothing=0. if exp == 'T00' else .1,
                images_dir=str(data/'images'), labels_dir=str(data/'labels'),
                out_dir=str(root/'runs'), pred_dir=str(root/'predictions'),
                curves_dir=str(root/'curves'), project_dir=str(root),
                save_test_predictions=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root, data = args.root.resolve(), args.data.resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    status_path = root/'cpu_status.json'
    jobs = configs(root, data, args.workers)
    manifest = {
        'driver_sha256': sha256(__file__), 'code_hashes': code_hashes(),
        'configs': [asdict(c) for c in jobs], 'environment': environment(),
        'threads': args.threads, 'interop_threads': 1,
        'processor': os.environ.get('PROCESSOR_IDENTIFIER', platform.processor()),
        'data_hashes': {n: sha256(data/'labels'/n) for n in
                        ('labels.csv','train_subset0.csv','val_subset0.csv','test_subset0.csv')},
        'selection': {'backbone': 'B03', 'recipe': 'T04',
                      'source': 'Colab seed0 validation screening; no test opened'},
        'policy': 'Six separate 10-epoch runs; CPU FP32; no test until all six complete and inference locked',
        'historical_runs_preserved': 11,
        'student_id': '2A202602521', 'student_name': 'HoangTrungAnh',
    }
    plan = root/'cpu_training_plan.json'
    if plan.exists():
        if read_json(plan) != manifest:
            raise ValueError('CPU driver/configuration/data/environment changed. Restore the saved plan before resume.')
    else:
        write_json(plan, manifest)
    splits = load_split(data/'labels')
    checks = check_split(*splits, data/'images')
    if [len(d) for d in splits] != [10501,3501,3507] or checks.get('missing', 0):
        raise ValueError(f'Invalid original fold0 data: {checks}')
    write_json(root/'cpu_split_checks.json', checks)
    print('CPU', manifest['processor'], 'threads', args.threads, flush=True)
    if not (root/'pipeline_checks.json').exists():
        evidence = pipeline_checks(data, root, device='cpu')
        if not evidence['passed']:
            raise RuntimeError('Real-image pipeline checks failed')
    if args.check_only:
        print('CPU data and pipeline checks passed', flush=True)
        return

    original_epoch = train.train_one_epoch
    current = {}

    def monitored_epoch(model, loader, *pos, **kw):
        current['epoch'] += 1
        def report(batch, total):
            write_json(status_path, {**current, 'phase': 'training',
                       'batch': batch, 'batches_per_epoch': total,
                       'pid': os.getpid(), 'updated_unix': time.time(),
                       'ready_to_submit': False})
            print(f"{current['exp_id']} seed {current['seed']} epoch {current['epoch']}/10 batch {batch}/{total}", flush=True)
        return original_epoch(model, ProgressLoader(loader, report), *pos, **kw)

    train.train_one_epoch = monitored_epoch
    try:
        for cfg in jobs:
            folder = train.run_dir(cfg)
            checkpoint = folder/'last.pt'
            start = 0
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                start = saved['epoch']+1
                del saved
            current.update(exp_id=cfg.exp_id, seed=cfg.seed, epoch=start)
            write_json(status_path, {**current, 'phase': 'loading_model',
                       'pid': os.getpid(), 'updated_unix': time.time(), 'ready_to_submit': False})
            print('Starting/resuming', cfg.exp_id, 'seed', cfg.seed, flush=True)
            train.run(cfg)
            gc.collect()
        write_json(status_path, {'phase': 'six_trainings_complete_awaiting_inference_and_test',
                   'completed_local_runs': 6, 'completed_main_runs': 17,
                   'ready_to_submit': False, 'updated_unix': time.time()})
    except BaseException as exc:
        write_json(status_path, {**current, 'phase': 'failed', 'error': str(exc),
                   'updated_unix': time.time(), 'ready_to_submit': False})
        traceback.print_exc()
        raise
    finally:
        train.train_one_epoch = original_epoch


if __name__ == '__main__':
    main()
