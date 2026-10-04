from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eval import read_pred, save_predictions
from .common import write_json
from .dataset import IMAGENET_MEAN, IMAGENET_STD, DeepWeedsDataset, build_transforms, load_split
from .losses import FocalLoss, LabelSmoothingCE, mix_batch
from .model import build_model, frozen_train_mode, param_groups
from .train import set_seed


def pipeline_checks(data_root, study_root, device='cuda', max_steps=400):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    study_root=Path(study_root)
    if (study_root/'final_lock.json').exists():
        raise RuntimeError('Do not repeat development checks after final lock')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Enable GPU before the pipeline overfit check')
    set_seed(0)
    train,_,_=load_split(Path(data_root)/'labels')
    subset=train.groupby('Label',sort=True).head(1).iloc[:4]
    dataset=DeepWeedsDataset(subset,Path(data_root)/'images',build_transforms(False,64))
    samples=[dataset[i] for i in range(4)]
    x=torch.stack([s[0] for s in samples]).to(device)
    y=torch.tensor([s[1] for s in samples],device=device)
    # Real backbone, four real train images; no pretrained downloads, no random augmentation.
    model=build_model('resnet18',pretrained=False,init='scratch').to(device)
    model.eval()
    with torch.no_grad():
        z=model(x)
    assert z.shape==(4,9)
    initial=float(nn.functional.cross_entropy(z,y))
    criterion=nn.CrossEntropyLoss()
    torch.testing.assert_close(FocalLoss(0)(z,y),criterion(z,y),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(LabelSmoothingCE(0)(z,y),criterion(z,y))
    model.train(); optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    history=[]
    for step in range(max_steps):
        optimizer.zero_grad(); loss=criterion(model(x),y); loss.backward(); optimizer.step()
        history.append(float(loss.detach()))
        if step%25==0:
            print('Overfit step',step,'loss',history[-1],flush=True)
        if history[-1]<.05:
            break
    passed=history[-1]<.05
    folder=study_root/'pipeline'; folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6,4)); ax.plot(history); ax.set_xlabel('Step'); ax.set_ylabel('CE on one fixed batch')
    fig.tight_layout(); fig.savefig(folder/'overfit_batch.png',dpi=160); plt.close(fig)
    # Visual verification includes ordinary augmentation and CutMix with its actual lambda.
    augmented=DeepWeedsDataset(subset,Path(data_root)/'images',build_transforms(True,224))
    images=torch.stack([augmented[i][0] for i in range(4)])
    mixed,targets=mix_batch(images,torch.tensor(subset.Label.values),mode='cutmix')
    fig,axes=plt.subplots(2,4,figsize=(12,6))
    mean=torch.tensor(IMAGENET_MEAN)[:,None,None]; std=torch.tensor(IMAGENET_STD)[:,None,None]
    for row,source in enumerate([images,mixed]):
        for i in range(4):
            axes[row,i].imshow((source[i]*std+mean).clamp(0,1).permute(1,2,0).numpy())
            axes[row,i].set_title(f'label={int(targets[0][i])}, paired={int(targets[1][i])}, λ={targets[2]:.3f}' if row else f'label={int(subset.Label.iloc[i])}')
            axes[row,i].axis('off')
    fig.tight_layout(); fig.savefig(folder/'augmentation_cutmix.png',dpi=160); plt.close(fig)
    checks={'passed':passed,'initial_ce':initial,'reference_ln9':float(np.log(9)),
            'initial_ce_note':'Diagnostic reference, not an equality guarantee with real non-uniform images',
            'overfit_loss':history[-1],'overfit_steps':len(history),'train_filenames':subset.Filename.tolist(),
            'device':device,'threshold':.05,'focal_gamma0_equals_ce':True,'smoothing0_equals_ce':True}
    write_json(study_root/'pipeline_checks.json',checks)
    del model,optimizer,x,y
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if not passed:
        raise RuntimeError('Overfit check failed; inspect pipeline before starting experiments')
    return checks
