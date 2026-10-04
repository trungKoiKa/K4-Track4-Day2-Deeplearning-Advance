from __future__ import annotations

import copy
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=1, keepdims=True)


def view_identity(x):
    return x


def view_hflip(x):
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop):
    h, w = x.shape[-2:]
    if crop > min(h, w):
        raise ValueError('crop is larger than input')
    offsets = [(0, 0), (0, w-crop), (h-crop, 0), (h-crop, w-crop), ((h-crop)//2, (w-crop)//2)]
    return [x[..., y:y+crop, z:z+crop] for y, z in offsets]


def views_multiscale(x, sizes):
    return [F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False) for s in sizes]


def aggregate_views(logits_per_view, space='prob'):
    if not logits_per_view:
        raise ValueError('at least one view required')
    if space == 'prob':
        return np.mean([softmax(z) for z in logits_per_view], axis=0)
    if space == 'logit':
        return softmax(np.mean(logits_per_view, axis=0))
    raise ValueError(space)


def ensemble_probs(list_of_probs):
    if not list_of_probs or len({p.shape for p in list_of_probs}) != 1:
        raise ValueError('matching arrays on the same filename order required')
    p = np.mean(list_of_probs, axis=0)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError('invalid probabilities')
    return p


def fit_temperature(val_logits, val_labels):
    logits = torch.as_tensor(val_logits, dtype=torch.float64)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=.1, max_iter=60, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.clamp(-4.6, 4.6).exp(), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    t = float(log_t.detach().clamp(-4.6, 4.6).exp())
    # A failed optimizer must not worsen the objective compared with T=1.
    if not np.isfinite(t) or F.cross_entropy(logits/t, labels) > F.cross_entropy(logits, labels):
        return 1.
    return t


def apply_temperature(logits, T):
    if not np.isfinite(T) or T <= 0:
        raise ValueError('temperature must be positive and finite')
    return softmax(np.asarray(logits) / T)


def pipeline_logits(model, x, method='single', amp=False, crop=224):
    if method == 'single':
        views = [x]
    elif method == 'hflip':
        views = [x, view_hflip(x)]
    elif method == 'fivecrop':
        views = views_multicrop(x, crop)
    else:
        raise ValueError(method)
    context = torch.autocast('cuda', dtype=torch.float16) if amp and x.is_cuda else nullcontext()
    with context:
        logits = [model(v).float() for v in views]
    return torch.stack(logits).mean(0)


def predict_logits(model, loader, device, view=None):
    model.eval()
    names, labels, logits = [], [], []
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            if view:
                x = view(x)
            z = model(x).float()
            names.extend(f); labels.extend(y.tolist()); logits.append(z.cpu().numpy())
    return names, np.asarray(labels, dtype=np.int64), np.concatenate(logits)


def predict_pipeline(model, loader, device, method='single', amp=False, crop=224):
    model.eval()
    names, labels, logits = [], [], []
    with torch.inference_mode():
        for x, y, f in loader:
            z = pipeline_logits(model, x.to(device, non_blocking=True), method, amp, crop)
            names.extend(f); labels.extend(y.tolist()); logits.append(z.cpu().numpy())
    return names, np.asarray(labels, dtype=np.int64), np.concatenate(logits)


def fuse_conv_bn(model):
    model = copy.deepcopy(model).eval()
    def walk(module):
        children = list(module.named_children())
        for name, child in children:
            walk(child)
        # Only architecturally verified patterns, never arbitrary sibling pairs.
        pairs = []
        if isinstance(module, nn.Sequential):
            pairs += [(children[i][0], children[i+1][0]) for i in range(len(children)-1)]
        for conv, bn in [('conv', 'bn'), ('conv1', 'bn1'), ('conv2', 'bn2'), ('conv3', 'bn3')]:
            if hasattr(module, conv) and hasattr(module, bn):
                pairs.append((conv, bn))
        for conv_name, bn_name in pairs:
            conv, bn = getattr(module, conv_name), getattr(module, bn_name)
            # timm BatchNormAct2d also contains an activation: do not silently delete it.
            if isinstance(conv, nn.Conv2d) and type(bn) is nn.BatchNorm2d:
                setattr(module, conv_name, torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn))
                setattr(module, bn_name, nn.Identity())
    walk(model)
    return model
