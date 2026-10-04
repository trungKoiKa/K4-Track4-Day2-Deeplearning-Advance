from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import environment, sha256, write_json
from .dataset import CLASS_NAMES, check_split, load_split

EXPECTED_MD5 = 'b7b30f96d466fba86016aa5a26606e0f'


def prepare_data(drive_root, data_root):
    drive_root, data_root = Path(drive_root), Path(data_root)
    data_root.mkdir(parents=True,exist_ok=True)
    labels_dir=data_root/'labels'; labels_dir.mkdir(exist_ok=True)
    for name in ('labels.csv','train_subset0.csv','val_subset0.csv','test_subset0.csv'):
        source=drive_root/'labels'/name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source,labels_dir/name)
    archive=data_root/'images.zip'
    def md5(path):
        h=hashlib.md5()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):
                h.update(chunk)
        return h.hexdigest()
    if not archive.exists() or md5(archive)!=EXPECTED_MD5:
        saved=drive_root/'images.zip'
        if saved.exists():
            shutil.copy2(saved,archive)
        else:
            print('Downloading original images.zip from Zenodo...',flush=True)
            temporary=archive.with_suffix('.download')
            urllib.request.urlretrieve('https://zenodo.org/records/7939060/files/images.zip?download=1',temporary)
            temporary.replace(archive)
    if md5(archive)!=EXPECTED_MD5:
        raise ValueError('MD5 mismatch; do not proceed')
    if not (drive_root/'images.zip').exists():
        shutil.copy2(archive,drive_root/'images.zip')
    images=data_root/'images'; images.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        entries=[m for m in z.infolist() if not m.is_dir()]
        if len(entries)!=17509:
            raise ValueError('Expected 17509 archive entries')
        for entry in entries:
            target=(images/entry.filename).resolve()
            if not target.is_relative_to(images.resolve()):
                raise ValueError('Unsafe archive path')
            if not target.exists() or target.stat().st_size!=entry.file_size:
                z.extract(entry,images)
    return data_root


def eda(data_root, output_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    data_root, output_root=Path(data_root),Path(output_root)
    output_root.mkdir(parents=True,exist_ok=True)
    train,val,test=load_split(data_root/'labels')
    checks=check_split(train,val,test,data_root/'images')
    all_labels=pd.read_csv(data_root/'labels'/'labels.csv')
    truth=all_labels.set_index('Filename').Label
    union=set(train.Filename)|set(val.Filename)|set(test.Filename)
    if union!=set(all_labels.Filename):
        raise ValueError('Split union differs from labels.csv')
    mismatches=[]
    for split,df in [('train',train),('val',val),('test',test)]:
        for row in df.itertuples():
            if int(truth[row.Filename])!=row.Label:
                mismatches.append({'split':split,'Filename':row.Filename,'split_label':row.Label,'labels_csv_label':int(truth[row.Filename])})
    checks['label_mismatches']=mismatches
    checks['csv_sha256']={p.name:sha256(p) for p in (data_root/'labels').glob('*.csv')}
    write_json(output_root/'split_checks.json',checks)
    counts=pd.DataFrame({'Species':CLASS_NAMES,'Total':all_labels.Label.value_counts().reindex(range(9)).values,
                        **{s:df.Label.value_counts().reindex(range(9)).values for s,df in [('train',train),('val',val),('test',test)]}})
    counts.to_csv(output_root/'class_counts.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    for ax,s in zip(axes,('train','val','test')):
        ax.bar(counts.Species,counts[s]); ax.set_title(s); ax.tick_params(axis='x',rotation=90); ax.set_ylabel('Images')
    fig.tight_layout(); fig.savefig(output_root/'class_distribution.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(9,3,figsize=(12,30))
    sample_stats=[]
    for label in range(9):
        selected=train[train.Label==label].sample(3,random_state=0)
        for j,row in enumerate(selected.itertuples()):
            with Image.open(data_root/'images'/row.Filename) as im:
                sample_stats.append({'Filename':row.Filename,'size':im.size,'mode':im.mode})
                axes[label,j].imshow(im.convert('RGB'))
            axes[label,j].set_title(CLASS_NAMES[label]+'\n'+row.Filename,fontsize=9); axes[label,j].axis('off')
    fig.tight_layout(); fig.savefig(output_root/'train_samples_3_per_class.png',dpi=120); plt.close(fig)
    write_json(output_root/'sample_image_stats.json',sample_stats)
    # Full-image metadata audit, not just the 27 samples.
    size_counts={}; mode_counts={}
    for filename in all_labels.Filename:
        with Image.open(data_root/'images'/filename) as im:
            size=str(im.size); size_counts[size]=size_counts.get(size,0)+1
            mode_counts[im.mode]=mode_counts.get(im.mode,0)+1
    write_json(output_root/'image_metadata.json',{'sizes':size_counts,'modes':mode_counts})
    write_json(output_root/'environment.json',environment())
    print(counts.to_string(index=False)); print('EDA saved:',output_root)
    return checks
