from __future__ import annotations

import argparse
import gc
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from eval import compute_metrics, save_predictions
from .common import atomic_torch_save, code_hashes, environment, read_json, sha256, write_json
from .dataset import build_transforms, check_split, load_split, make_loader
from .inference import softmax
from .losses import build_criterion, class_weights, mix_batch, mixed_loss
from .model import build_model, count_params, frozen_train_mode, param_groups, profile_gmacs


@dataclass
class Config:
    exp_id: str = 'T00'
    seed: int = 0
    fold: int = 0
    backbone: str = 'resnet50'
    pretrained_tag: str | None = None
    init: str = 'finetune'
    drop_rate: float = 0.
    img_size: int = 224
    aug: str = 'basic'
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.
    loss: str = 'ce'
    label_smoothing: float = 0.
    focal_gamma: float = 2.
    class_weight_beta: float | None = None
    epochs: int = 12  # Original public default; the experiment plan explicitly sets 10.
    batch_size: int = 64
    grad_accum: int = 1
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = .05
    warmup_epochs: float = 1.
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = 'data/images'
    labels_dir: str = 'data/labels'
    out_dir: str = 'runs'
    pred_dir: str = 'predictions'
    curves_dir: str = 'curves'
    project_dir: str = '.'
    device: str = 'cuda'
    resume: bool = True
    profile: bool = True
    save_test_predictions: bool = False


def run_dir(cfg):
    return Path(cfg.out_dir) / cfg.exp_id / f'seed{cfg.seed}'


def pred_path(cfg, split):
    return Path(cfg.pred_dir) / f'{cfg.exp_id}_seed{cfg.seed}_{split}.csv'


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_optimizer(model, cfg):
    return torch.optim.AdamW(param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = round(cfg.warmup_epochs * steps_per_epoch)
    def scale(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        fraction = min(1., max(0., (step-warmup) / max(1, total-warmup)))
        return .5 * (1 + math.cos(math.pi*fraction))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class EMA:
    def __init__(self, model, decay):
        import copy
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for name, value in self.model.state_dict().items():
            source = model.state_dict()[name]
            # Copy BN buffers, average only trainable floating weights.
            if name in dict(model.named_parameters()) and value.is_floating_point():
                value.mul_(self.decay).add_(source, alpha=1-self.decay)
            else:
                value.copy_(source)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None):
    model.train()
    if cfg.init == 'frozen':
        frozen_train_mode(model)
    optimizer.zero_grad(set_to_none=True)
    total_loss, n, optimizer_steps = 0., 0, 0
    for step, (x, y, _) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        targets = None
        if cfg.mix:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        context = torch.autocast('cuda', dtype=torch.float16) if cfg.amp and x.is_cuda else nullcontext()
        # Correct denominator for the last, partial accumulation window.
        window_start = (step // cfg.grad_accum) * cfg.grad_accum
        window_length = min(cfg.grad_accum, len(loader)-window_start)
        with context:
            logits = model(x)
            loss = mixed_loss(criterion, logits, targets) if targets else criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        scaler.scale(loss / window_length).backward()
        total_loss += float(loss.detach()) * len(y); n += len(y)
        if (step+1) % cfg.grad_accum == 0 or step+1 == len(loader):
            previous_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            applied = scaler.get_scale() >= previous_scale
            if applied:
                scheduler.step()
                optimizer_steps += 1
                if ema:
                    ema.update(model)
            optimizer.zero_grad(set_to_none=True)
    return {'train_loss': total_loss/n, 'lr': optimizer.param_groups[0]['lr'], 'optimizer_steps': optimizer_steps}


def evaluate(model, loader, criterion, device):
    model.eval()
    names, labels, logits = [], [], []
    total, n = 0., 0
    with torch.inference_mode():
        for x, y, f in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            z = model(x).float()
            loss = criterion(z, y)
            total += float(loss) * len(y); n += len(y)
            names.extend(f); labels.extend(y.cpu().tolist()); logits.append(z.cpu().numpy())
    return names, np.asarray(labels, dtype=np.int64), np.concatenate(logits), total/n


def plot_curves(history, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(frame.epoch, frame.train_loss, label='train (recipe loss)')
    axes[0].plot(frame.epoch, frame.val_loss, label='val (plain CE)')
    axes[1].plot(frame.epoch, frame.macro_f1, label='val macro-F1')
    axes[1].plot(frame.epoch, frame.top1, label='val top-1')
    axes[2].plot(frame.epoch, frame.lr, label='backbone LR')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
    axes[0].set_ylabel('Loss'); axes[1].set_ylabel('Metric'); axes[2].set_ylabel('LR')
    fig.suptitle(title); fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160); plt.close(fig)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def compatibility(cfg):
    ignored = {'images_dir', 'labels_dir', 'out_dir', 'pred_dir', 'curves_dir', 'project_dir', 'resume', 'profile', 'save_test_predictions'}
    return {k:v for k,v in asdict(cfg).items() if k not in ignored}


def run(cfg):
    if cfg.fold != 0:
        raise ValueError('This required-track runner only permits fold 0')
    if cfg.epochs < 1 or cfg.batch_size < 2 or cfg.grad_accum < 1:
        raise ValueError('invalid training budget')
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Enable a Colab GPU before training; EDA works on CPU')
    if cfg.save_test_predictions and not (Path(cfg.project_dir)/'final_lock.json').is_file():
        raise RuntimeError('Test is locked until final_lock.json is created from val')
    folder = run_dir(cfg); folder.mkdir(parents=True, exist_ok=True)
    data_hashes = {n:sha256(Path(cfg.labels_dir)/n) for n in ('labels.csv','train_subset0.csv','val_subset0.csv','test_subset0.csv')}
    meta = {'compatibility':compatibility(cfg), 'data_hashes':data_hashes, 'code_hashes':code_hashes()}
    if (folder/'metadata.json').exists():
        old = read_json(folder/'metadata.json')
        if old != meta:
            raise ValueError('Resume configuration/data/code changed. Preserve this run; use a new exp_id.')
        if (folder/'result.json').exists():
            print('Already complete:', cfg.exp_id, cfg.seed)
            if cfg.save_test_predictions:
                from .experiments import final_predict
                final_predict(cfg)
            return read_json(folder/'result.json')
    else:
        write_json(folder/'metadata.json',meta)
        write_json(folder/'config.json',asdict(cfg)); write_json(folder/'environment.json',environment())
    set_seed(cfg.seed)
    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    write_json(folder/'split_checks.json',check_split(train_df,val_df,test_df,cfg.images_dir))
    train_loader = make_loader(train_df,cfg.images_dir,build_transforms(True,cfg.img_size,cfg.aug),cfg.batch_size,True,cfg.sampler,cfg.num_workers,cfg.seed)
    val_loader = make_loader(val_df,cfg.images_dir,build_transforms(False,cfg.img_size),cfg.batch_size,False,num_workers=cfg.num_workers)
    checkpoint = folder/'last.pt'
    if checkpoint.exists() and not cfg.resume:
        raise FileExistsError('Existing checkpoint; use resume rather than overwrite')
    model = build_model(cfg.pretrained_tag or cfg.backbone, pretrained=not checkpoint.exists(),drop_rate=cfg.drop_rate,init=cfg.init).to(cfg.device)
    if checkpoint.exists():
        pretrained_info = read_json(folder/'pretrained.json')
    else:
        pretrained_info = model.pretrained_cfg
        write_json(folder/'pretrained.json',pretrained_info)
    if tuple(pretrained_info.get('mean', (.485,.456,.406))) != (.485,.456,.406) or tuple(pretrained_info.get('std',(.229,.224,.225))) != (.229,.224,.225):
        raise ValueError('Selected pretrained tag is not compatible with shared ImageNet normalization')
    params = count_params(model)
    if cfg.profile and not (folder/'profile.json').exists():
        write_json(folder/'profile.json',profile_gmacs(model,cfg.img_size))
    criterion = build_criterion(cfg.loss,smoothing=cfg.label_smoothing,gamma=cfg.focal_gamma,
                 weight=class_weights(train_df.Label.value_counts().reindex(range(9)).values,cfg.class_weight_beta or 0)).to(cfg.device)
    optimizer = build_optimizer(model,cfg)
    scheduler = build_scheduler(optimizer,cfg,math.ceil(len(train_loader)/cfg.grad_accum))
    scaler = torch.amp.GradScaler('cuda',enabled=cfg.amp and cfg.device.startswith('cuda'))
    ema = EMA(model,cfg.ema_decay) if cfg.ema_decay is not None else None
    history, start_epoch, best_score, best_epoch = [], 0, -1., -1
    if checkpoint.exists():
        state = torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        if ema:
            ema.model.load_state_dict(state['ema'])
        history, start_epoch = state['history'], state['epoch']+1
        best_score, best_epoch = state['best_score'], state['best_epoch']
        restore_rng(state['rng'])
    for epoch in range(start_epoch,cfg.epochs):
        train_loader.generator.manual_seed(cfg.seed+epoch)
        if cfg.device.startswith('cuda'):
            torch.cuda.synchronize()
        started = time.perf_counter()
        stats = train_one_epoch(model,train_loader,criterion,optimizer,scheduler,scaler,cfg,cfg.device,ema)
        if cfg.device.startswith('cuda'):
            torch.cuda.synchronize()
        train_seconds = time.perf_counter()-started
        evaluated = ema.model if ema else model
        names, labels, logits, val_loss = evaluate(evaluated,val_loader,nn.CrossEntropyLoss(),cfg.device)
        probs = softmax(logits); metrics = compute_metrics(labels,probs.argmax(1),probs)
        history.append({'epoch':epoch+1,**stats,'val_loss':val_loss,'macro_f1':metrics['macro_f1'],
                        'top1':metrics['top1'],'train_seconds':train_seconds,'epoch_seconds':time.perf_counter()-started})
        improved = metrics['macro_f1'] > best_score
        if improved:
            best_score, best_epoch = metrics['macro_f1'], epoch+1
        state = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                 'scaler':scaler.state_dict(),'ema':ema.model.state_dict() if ema else None,'epoch':epoch,
                 'history':history,'best_score':best_score,'best_epoch':best_epoch,'rng':rng_state(),
                 'metadata':meta,'training_model':model.state_dict(),'evaluation_model':evaluated.state_dict()}
        if improved:
            atomic_torch_save({**state,'model':evaluated.state_dict(),'macro_f1':best_score,
                              'epoch_number':epoch+1},folder/'best.pt')
        atomic_torch_save(state,checkpoint)
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        plot_curves(history,Path(cfg.curves_dir)/f'{cfg.exp_id}_seed{cfg.seed}.png',f'{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}')
        print(f'{cfg.exp_id} seed {cfg.seed} epoch {epoch+1}/{cfg.epochs}: F1={metrics["macro_f1"]:.4f}, train={train_seconds:.1f}s',flush=True)
    best = torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(best['model'])
    names, labels, logits, _ = evaluate(model,val_loader,nn.CrossEntropyLoss(),cfg.device)
    np.savez_compressed(folder/'val_logits.npz',filenames=np.asarray(names),y_true=labels,logits=logits)
    save_predictions(pred_path(cfg,'val'),names,labels,softmax(logits))
    metrics = compute_metrics(labels,softmax(logits).argmax(1),softmax(logits))
    result = {'exp_id':cfg.exp_id,'seed':cfg.seed,'backbone':cfg.backbone,'pretrained_tag':pretrained_info.get('tag'),
              'best_epoch':best_epoch,'params_m':params,'train_seconds_per_epoch':float(np.mean([h['train_seconds'] for h in history])),
              'epochs':cfg.epochs,'img_size':cfg.img_size,'macro_f1':metrics['macro_f1'],'top1':metrics['top1'],
              'ece':metrics['ece'],'balanced_acc':metrics['balanced_acc'],'n':metrics['n']}
    if cfg.profile:
        result['gmac'] = read_json(folder/'profile.json')['gmac']
    if cfg.save_test_predictions:
        from .experiments import final_predict
        final_predict(cfg, model=model)
    write_json(folder/'result.json',result)
    del model, optimizer, ema, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_overrides(pairs):
    defaults = asdict(Config())
    annotations = {f.name:str(f.type) for f in fields(Config)}
    values = {}
    for pair in pairs:
        key, sep, value = pair.partition('=')
        if not sep or key not in defaults:
            raise ValueError(f'Unknown override {pair}')
        if value.lower() in ('none','null'):
            if 'None' not in annotations[key]:
                raise ValueError(f'{key} cannot be None')
            values[key] = None
        elif 'bool' in annotations[key]:
            if value.lower() not in ('true','false'):
                raise ValueError(f'{key}: expected true/false')
            values[key] = value.lower() == 'true'
        elif 'int' in annotations[key]:
            values[key] = int(value)
        elif 'float' in annotations[key]:
            values[key] = float(value)
        else:
            values[key] = value
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set',nargs='*',default=[])
    args = parser.parse_args()
    print(run(Config(**parse_overrides(args.set))))


if __name__ == '__main__':
    main()
