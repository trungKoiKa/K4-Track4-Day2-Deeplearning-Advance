from __future__ import annotations

import copy
import time

import numpy as np
import torch

from .inference import pipeline_logits


def bench(fn, warmup=10, iters=100, sync=None):
    if warmup < 10 or iters < 50:
        raise ValueError('at least 10 warmup and 50 measured iterations required')
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        sync(); start = time.perf_counter(); fn(); sync()
        times.append((time.perf_counter() - start)*1000)
    return {**{f'p{p}': float(np.percentile(times, p)) for p in (50, 95, 99)},
            'mean': float(np.mean(times)), 'n': iters, 'warmup': warmup, 'samples_ms': times}


def pipeline_latency(model, batch_size=1, img_size=224, method='single', dtype='fp32', temperature=1.,
                     device='cuda', warmup=10, iters=100):
    if dtype not in ('fp32', 'amp', 'fp16'):
        raise ValueError(dtype)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('GPU required for lab latency measurements')
    model = copy.deepcopy(model).to(device).eval()
    size = 256 if method == 'fivecrop' else img_size
    x = torch.randn(batch_size, 3, size, size, device=device)
    if dtype == 'fp16':
        model.half(); x = x.half()
    def forward():
        logits = pipeline_logits(model, x, method, dtype == 'amp', img_size)
        return torch.softmax(logits / temperature, dim=1)
    with torch.inference_mode():
        result = bench(forward, warmup, iters, torch.cuda.synchronize if x.is_cuda else None)
    result.update({'gpu': torch.cuda.get_device_name(0) if x.is_cuda else 'CPU', 'dtype': dtype,
                   'batch': batch_size, 'img_size': img_size, 'method': method, 'temperature': temperature,
                   'fused_bn': False, 'images_per_s': batch_size/(result['p50']/1000),
                   'torch': torch.__version__, 'scope': 'GPU forward + view aggregation + softmax/calibration; excludes disk/CPU preprocessing'})
    del model, x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def latency_report(model, batch_size, img_size, dtype='fp32', device='cuda', warmup=10, iters=100):
    return pipeline_latency(model, batch_size, img_size, dtype=dtype, device=device, warmup=warmup, iters=iters)


def tta_latency(model, k_views, **kw):
    if k_views not in (1, 2, 5):
        raise ValueError('implemented view counts: 1,2,5')
    return pipeline_latency(model, method={1:'single', 2:'hflip', 5:'fivecrop'}[k_views], **kw)
