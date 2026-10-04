from __future__ import annotations

import copy
import timm
import torch
from torch import nn

SUGGESTED_BACKBONES = {'resnet50': 'resnet50', 'resnext50': 'resnext50_32x4d',
                       'convnext_tiny': 'convnext_tiny', 'deit_small': 'deit_small_patch16_224',
                       'mobilenetv3': 'mobilenetv3_large_100'}


def build_model(name, pretrained=True, num_classes=9, drop_rate=0., init='finetune'):
    if init not in ('scratch', 'frozen', 'finetune'):
        raise ValueError(init)
    # Disable fused attention for transparent MAC accounting and deterministic profiling.
    timm.layers.set_fused_attn(False)
    model = timm.create_model(name, pretrained=pretrained and init != 'scratch',
                              num_classes=num_classes, drop_rate=drop_rate)
    if init == 'frozen':
        freeze_backbone(model)
    return model


def freeze_backbone(model):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in head_ids)
    model.eval()
    model.get_classifier().train()


def frozen_train_mode(model):
    model.eval()
    model.get_classifier().train()


def param_groups(model, lr_backbone, lr_head, weight_decay):
    # Four groups: the head's norm/bias must also be exempt from decay.
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    no_decay = set(model.no_weight_decay()) if hasattr(model, 'no_weight_decay') else set()
    norm_types = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm,
                  nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    norm_ids = {id(p) for m in model.modules() if isinstance(m, norm_types) for p in m.parameters(recurse=False)}
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr = lr_head if id(p) in head_ids else lr_backbone
        decay = 0. if p.ndim <= 1 or name.endswith('.bias') or id(p) in norm_ids or name in no_decay else weight_decay
        groups.setdefault((lr, decay), []).append(p)
    return [{'params': ps, 'lr': lr, 'weight_decay': decay} for (lr, decay), ps in groups.items()]


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def profile_gmacs(model, img_size=224):
    from fvcore.nn import FlopCountAnalysis
    clone = copy.deepcopy(model).cpu().eval()
    analysis = FlopCountAnalysis(clone, torch.zeros(1, 3, img_size, img_size))
    analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    total = analysis.total()
    return {'gmac': total / 1e9, 'tool': 'fvcore (one multiply-add counted as one)',
            'unsupported_ops': dict(analysis.unsupported_ops()), 'by_operator': dict(analysis.by_operator())}


def count_gmacs(model, img_size=224):
    profile = profile_gmacs(model, img_size)
    if profile['unsupported_ops']:
        print('MAC accounting exclusions:', profile['unsupported_ops'])
    return profile['gmac']
