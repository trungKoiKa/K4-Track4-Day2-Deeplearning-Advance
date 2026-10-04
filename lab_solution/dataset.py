from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
CLASS_NAMES = ['Chinee apple', 'Lantana', 'Parkinsonia', 'Parthenium', 'Prickly acacia',
               'Rubber vine', 'Siam weed', 'Snake weed', 'Negative']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir, fold=0):
    frames = []
    for split in ('train', 'val', 'test'):
        df = pd.read_csv(Path(labels_dir) / f'{split}_subset{fold}.csv')
        if not {'Filename', 'Label'}.issubset(df.columns):
            raise ValueError(f'{split}: required columns Filename, Label')
        df['Filename'] = df['Filename'].astype(str)
        values = pd.to_numeric(df['Label'], errors='raise')
        if not np.all(values == values.astype(int)) or not values.between(0, 8).all():
            raise ValueError(f'{split}: labels must be integers 0..8')
        df['Label'] = values.astype(int)
        frames.append(df)
    return tuple(frames)


def check_split(train_df, val_df, test_df, images_dir):
    frames = dict(zip(('train', 'val', 'test'), (train_df, val_df, test_df)))
    sets = {s: set(df.Filename) for s, df in frames.items()}
    counts = {s: len(df) for s, df in frames.items()}
    for s, df in frames.items():
        if len(sets[s]) != len(df):
            raise ValueError(f'Duplicate filename in {s}')
        if set(df.Label) != set(range(9)):
            raise ValueError(f'{s}: all nine classes are required')
        expected = .6 if s == 'train' else .2
        if abs(len(df) / 17509 - expected) > .01:
            raise ValueError(f'{s}: split ratio exceeds tolerance; contact instructor')
    overlaps = {f'{a}_{b}': len(sets[a] & sets[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    if any(overlaps.values()):
        raise ValueError(f'Overlapping splits: {overlaps}')
    union = set.union(*sets.values())
    if len(union) != 17509:
        raise ValueError(f'Union must contain 17509 images, got {len(union)}')
    missing = [n for n in union if not (Path(images_dir) / n).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing images: {missing[:5]}')
    report = {'n': counts, 'union': len(union), 'overlap': overlaps, 'missing': len(missing),
              'per_class': {s: df.Label.value_counts().reindex(range(9), fill_value=0).to_dict() for s, df in frames.items()}}
    print(report)
    return report


def build_transforms(train, img_size=224, aug='basic'):
    if train:
        ops = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
        extras = {'basic': [], 'color': [T.ColorJitter(.2, .2, .2, .05)],
                  'trivial': [T.TrivialAugmentWide()], 'randaug': [T.RandAugment()]}
        if aug not in extras:
            raise ValueError(f'Unknown augmentation {aug}')
        ops += extras[aug]
    else:
        ops = [T.Resize(round(img_size * 256 / 224)), T.CenterCrop(img_size)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def inference_transforms(img_size=224, multicrop=False):
    if not multicrop:
        return build_transforms(False, img_size)
    # Preserve the full resized image; crop inside the GPU pipeline.
    return T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class DeepWeedsDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.images_dir / row.Filename) as im:
            image = im.convert('RGB')
            if self.transform:
                image = self.transform(image)
        return image, int(row.Label), str(row.Filename)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(df, images_dir, transform, batch_size, train, sampler=None, num_workers=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    sample = None
    if sampler == 'balanced':
        counts = df.Label.value_counts()
        weights = [1 / counts[int(y)] for y in df.Label]
        sample = WeightedRandomSampler(weights, len(df), replacement=True, generator=generator)
    elif sampler is not None:
        raise ValueError(f'Unknown sampler {sampler}')
    return DataLoader(DeepWeedsDataset(df, images_dir, transform), batch_size=batch_size,
                      shuffle=train and sample is None, sampler=sample, drop_last=train,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=generator, persistent_workers=False)
