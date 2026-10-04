from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LabelSmoothingCE(nn.CrossEntropyLoss):
    def __init__(self, smoothing=.1):
        super().__init__(label_smoothing=smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError('gamma must be nonnegative')
        self.gamma = gamma
        self.register_buffer('alpha', None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        log_pt = F.log_softmax(logits.float(), dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -(1 - log_pt.exp()).pow(self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def build_criterion(kind='ce', **kw):
    if kind == 'ce':
        return nn.CrossEntropyLoss()
    if kind == 'ls':
        return LabelSmoothingCE(kw.get('smoothing', .1))
    if kind == 'focal':
        return FocalLoss(kw.get('gamma', 2.), kw.get('alpha'))
    if kind == 'ce_weighted':
        return nn.CrossEntropyLoss(weight=kw['weight'])
    raise ValueError(kind)


def class_weights(counts, beta=0.):
    counts = torch.tensor(np.array(counts, copy=True), dtype=torch.float64)
    if (counts <= 0).any() or not 0 <= beta < 1:
        raise ValueError('positive train counts and beta in [0,1) required')
    w = counts.reciprocal() if beta == 0 else (1 - beta) / (1 - beta**counts)
    return (w / w.mean()).float()


def mix_batch(x, y, alpha=1., mode='cutmix'):
    if alpha <= 0:
        raise ValueError('alpha must be positive')
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], device=x.device)
    if mode == 'mixup':
        mixed = lam * x + (1 - lam) * x[perm]
    elif mode == 'cutmix':
        h, w = x.shape[-2:]
        ratio = np.sqrt(1 - lam)
        cut_w, cut_h = int(w * ratio), int(h * ratio)
        cx, cy = np.random.randint(w), np.random.randint(h)
        x1, x2 = max(0, cx - cut_w // 2), min(w, cx + (cut_w + 1) // 2)
        y1, y2 = max(0, cy - cut_h // 2), min(h, cy + (cut_h + 1) // 2)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (h * w)
    else:
        raise ValueError(mode)
    return mixed, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    a, b, lam = targets
    return lam * criterion(logits, a) + (1 - lam) * criterion(logits, b)
