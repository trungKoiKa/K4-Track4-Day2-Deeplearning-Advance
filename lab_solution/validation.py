"""Read-only acceptance checks; never equate export success with a completed lab."""
from pathlib import Path

import pandas as pd
import eval as ev
from .common import read_json, sha256, code_hashes


def validate_completion(root, data_root, notebook_url='', require_exports=True):
    root, data_root = Path(root), Path(data_root)
    errors=[]
    def need(path):
        path=root/path
        if not path.is_file() or path.stat().st_size==0:
            errors.append('Missing/empty: '+path.relative_to(root).as_posix()); return False
        return True
    def check_json(path, fn):
        if need(path):
            try:
                if not fn(read_json(root/path)): errors.append('Invalid: '+str(path))
            except Exception as exc: errors.append(str(path)+': '+str(exc))
    runs=[(f'B{i:02d}',0) for i in range(1,6)]+[(f'T{i:02d}',0) for i in range(1,7)]+[(e,s) for e in ('T00','F01') for s in range(3)]
    for exp,seed in runs:
        folder=Path('runs')/exp/f'seed{seed}'
        check_json(folder/'config.json',lambda r:r['epochs']==10 and r['seed']==seed and r['exp_id']==exp and r['fold']==0)
        check_json(folder/'result.json',lambda r:r['epochs']==10 and r['seed']==seed and r['exp_id']==exp)
        check_json(folder/'metadata.json',lambda r:r['code_hashes']==code_hashes())
        for name in ('best.pt','last.pt','environment.json','pretrained.json','profile.json','val_logits.npz'):
            need(folder/name)
        if need(folder/'history.csv'):
            try:
                df=pd.read_csv(root/folder/'history.csv')
                if df.epoch.tolist()!=list(range(1,11)) or not {'train_loss','val_loss','macro_f1','top1'}.issubset(df):
                    errors.append('Invalid epoch history: '+str(folder))
            except Exception as exc: errors.append(str(exc))
        need(Path('curves')/f'{exp}_seed{seed}.png')
        need(Path('predictions')/f'{exp}_seed{seed}_val.csv')
    for e in ('T00','F01'):
        for s in range(3):
            folder=Path('runs')/e/f'seed{s}'
            pred=Path('predictions')/f'{e}_seed{s}_test.csv'
            check_json(folder/'test_complete.json',lambda r,p=pred:(root/p).exists() and r['sha256']==sha256(root/p))
            for batch in (1,32):
                check_json(folder/f'final_latency_batch{batch}.json',lambda r,b=batch:r['batch']==b and r['n']>=100 and r['warmup']>=10 and r['gpu']!='CPU')
            for split in ('val','test'):
                p=Path('predictions')/f'{e}_seed{s}_{split}.csv'
                if need(p):
                    try: ev.check_against_csv(ev.read_pred(root/p),data_root/'labels'/f'{split}_subset0.csv')
                    except Exception as exc: errors.append(str(p)+': '+str(exc))
    for i in range(5):
        for batch in (1,32):
            check_json(Path('inference')/f'I{i:02d}_batch{batch}_latency.json',lambda r,b=batch:r['batch']==b and r['n']>=100 and r['warmup']>=10 and r['gpu']!='CPU')
        need(Path('inference')/f'I{i:02d}_val_logits.npz')
    check_json(Path('pipeline_checks.json'),lambda r:r['passed'] and r['overfit_loss']<.05)
    check_json(Path('eda/split_checks.json'),lambda r:r['n']=={'train':10501,'val':3501,'test':3507} and r['union']==17509 and not any(r['overlap'].values()) and r['missing']==0)
    for name in ('eda/class_counts.csv','eda/class_distribution.png','eda/train_samples_3_per_class.png','eda/image_metadata.json',
                 'pipeline/overfit_batch.png','pipeline/augmentation_cutmix.png','runtime_versions.json','environment.lock.txt',
                 'backbone_comparison.json','training_baseline.json','recipe_selection.json','inference_comparison.json','final_lock.json',
                 'analysis_notes.md','fresh_session_validation.json'):
        need(Path(name))
    if (root/'analysis_notes.md').exists() and len((root/'analysis_notes.md').read_text(encoding='utf-8').strip())<200:
        errors.append('analysis_notes.md: needs reviewed EDA/error analysis, >=200 characters')
    if not notebook_url.startswith('https://colab.research.google.com/'):
        errors.append('Missing runnable Colab notebook URL')
    check_json(Path('fresh_session_validation.json'),lambda r:r['session_id']!=r['initial_session_id'] and r['code_hashes']==code_hashes() and len(r['checked'])==6 and r['test_rerun'] is False)
    if (root/'final_lock.json').exists():
        lock=read_json(root/'final_lock.json')
        if lock.get('code_hashes')!=code_hashes(): errors.append('Final lock code hashes changed')
        if lock['pipeline']['calibrate']:
            for seed in range(3): need(Path('predictions')/f'F01uncal_seed{seed}_test.csv')
    if require_exports:
        for name in ('results.xlsx','report.md','README.md','export_provenance.json','analysis/backbone_latency.png',
                     'analysis/backbone_params.png','analysis/F01_confusion.png','analysis/error_images_seed0.png','inference/accuracy_latency.png'):
            need(Path(name))
    return {'ready_to_submit':not errors,'errors':errors,'expected_main_runs':len(runs)}
