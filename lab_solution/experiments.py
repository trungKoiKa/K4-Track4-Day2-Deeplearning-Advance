"""Study orchestration: val-only selection, immutable final configuration, resumable test inference."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import functools
from datetime import datetime, timezone
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from eval import compute_metrics, save_predictions, read_pred, check_against_csv
from .benchmark import pipeline_latency
from .common import atomic_torch_save, code_hashes, environment, read_json, sha256, write_json, SESSION_ID
from .dataset import build_transforms, inference_transforms, load_split, make_loader
from .inference import apply_temperature, fit_temperature, pipeline_logits, predict_pipeline, softmax
from .model import build_model, count_params
from .train import Config, compatibility, run, run_dir

BACKBONES = [('B01','resnet50'), ('B02','resnext50_32x4d'), ('B03','convnext_tiny'),
             ('B04','deit_small_patch16_224'), ('B05','mobilenetv3_large_100')]
PRETRAINED_TAGS = {'resnet50':'resnet50.a1_in1k','resnext50_32x4d':'resnext50_32x4d.a1h_in1k',
                  'convnext_tiny':'convnext_tiny.fb_in1k','deit_small_patch16_224':'deit_small_patch16_224.fb_in1k',
                  'mobilenetv3_large_100':'mobilenetv3_large_100.ra_in1k'}
CHANGES = {'T01':{'init':'scratch'}, 'T02':{'init':'frozen'}, 'T03':{'mix':'cutmix'},
           'T04':{'loss':'ls','label_smoothing':.1}, 'T05':{'loss':'focal','focal_gamma':2.}}
METHODS = {'I00':{'method':'single','amp':False}, 'I01':{'method':'hflip','amp':False},
           'I02':{'method':'fivecrop','amp':False}, 'I03':{'method':'single','amp':True},
           'I04':{'method':'single','amp':False}}


def load_trained(cfg, device='cuda'):
    model = build_model(cfg.pretrained_tag or cfg.backbone, pretrained=False, drop_rate=cfg.drop_rate, init=cfg.init)
    state = torch.load(run_dir(cfg)/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'])
    return model.to(device).eval()


def release(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def lock_digest(lock):
    return hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def record_failure(fn):
    @functools.wraps(fn)
    def wrapped(self,*args,**kwargs):
        try:
            return fn(self,*args,**kwargs)
        except (Exception,KeyboardInterrupt) as exc:
            path=self.root/'failures.json'
            entries=read_json(path) if path.exists() else []
            entries.append({'time_utc':datetime.now(timezone.utc).isoformat(),'operation':fn.__name__,
                            'args':list(args),'kwargs':kwargs,'type':type(exc).__name__,'message':str(exc)})
            write_json(path,entries)
            raise
    return wrapped


class Study:
    def __init__(self, root, data_root, notebook_url='', device='cuda'):
        self.root, self.data = Path(root), Path(data_root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.notebook_url, self.device = notebook_url, device
        if not (self.root/'initial_session.json').exists():
            write_json(self.root/'initial_session.json',{'session_id':SESSION_ID})

    def selection_open(self):
        if (self.root/'final_lock.json').exists():
            raise RuntimeError('Configuration is locked. Continue finals/export; do not revise selections.')

    def config(self, exp_id, backbone='resnet50', seed=0, **kw):
        batch = read_json(self.root/'batch_plan.json') if (self.root/'batch_plan.json').exists() else {'batch_size':16,'grad_accum':4}
        params = dict(exp_id=exp_id,seed=seed,backbone=backbone,epochs=10,**batch,
                      images_dir=str(self.data/'images'),labels_dir=str(self.data/'labels'),
                      out_dir=str(self.root/'runs'),pred_dir=str(self.root/'predictions'),
                      curves_dir=str(self.root/'curves'),project_dir=str(self.root),device=self.device)
        params.update(kw)
        return Config(**params)

    @record_failure
    def preflight(self):
        """Run before any B/T/F training. Determine a shared batch budget across all five models."""
        self.selection_open()
        if not torch.cuda.is_available():
            raise RuntimeError('Preflight requires a Colab GPU')
        if (self.root/'batch_plan.json').exists():
            old=read_json(self.root/'backbone_tags.json')
            if any(old[name]['name']!=tag for name,tag in PRETRAINED_TAGS.items()):
                raise ValueError('Existing preflight tags differ from the locked ImageNet-1k plan; do not mix studies')
            return read_json(self.root/'batch_plan.json')
        resolved = {}
        for batch, accum in ((16,4),(8,8)):
            passed = True
            for exp_id, name in BACKBONES:
                model = None
                try:
                    model = build_model(PRETRAINED_TAGS[name],pretrained=False).cuda()
                    pretrained = model.pretrained_cfg
                    tag = pretrained.get('tag')
                    tagged = PRETRAINED_TAGS[name]
                    resolved[name] = {'name':tagged,'pretrained_cfg':pretrained,'params_m':count_params(model)}
                    if tuple(pretrained.get('mean',())) != (.485,.456,.406) or tuple(pretrained.get('std',())) != (.229,.224,.225):
                        raise ValueError(f'{tagged}: incompatible normalization')
                    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
                    x = torch.randn(batch,3,224,224,device='cuda')
                    y = torch.randint(9,(batch,),device='cuda')
                    with torch.autocast('cuda',dtype=torch.float16):
                        z = model(x)
                        assert z.shape == (batch,9)
                        loss = torch.nn.functional.cross_entropy(z,y)
                    loss.backward()
                    optimizer.step()  # Include Adam state allocations in the memory feasibility check.
                    print('Preflight OK:',name,batch,flush=True)
                    del x,y,z,loss,optimizer
                except torch.cuda.OutOfMemoryError:
                    passed = False
                    print('OOM:',name,'batch',batch)
                finally:
                    # Drop partial allocations too when a forward/backward/optimizer step OOMs.
                    x=y=z=loss=optimizer=None
                    if model is not None:
                        del model
                    gc.collect(); torch.cuda.empty_cache()
                if not passed:
                    break
            if passed:
                plan = {'batch_size':batch,'grad_accum':accum}
                write_json(self.root/'batch_plan.json',plan)
                write_json(self.root/'backbone_tags.json',resolved)
                write_json(self.root/'preflight_environment.json',environment())
                return plan
        raise RuntimeError('Neither shared batch plan fits. Do not start partial comparisons.')

    def require_pipeline(self):
        if not (self.root/'pipeline_checks.json').exists() or not read_json(self.root/'pipeline_checks.json').get('passed'):
            raise RuntimeError('Run pipeline_checks before training')

    def check_environment(self):
        current=environment()
        path=self.root/'runtime_versions.json'
        keys=('python','torch','torchvision','timm','numpy','pandas','Pillow','matplotlib','fvcore','openpyxl')
        if path.exists():
            old=read_json(path)
            changes={k:(old.get(k),current.get(k)) for k in keys if old.get(k)!=current.get(k)}
            if changes:
                raise RuntimeError(f'Runtime versions changed; restore the recorded environment before resume: {changes}')
        else:
            write_json(path,current)
            lines=[f'{name}=={current[name]}' for name in keys if name!='python']
            (self.root/'environment.lock.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return current

    @record_failure
    def train_backbone(self, exp_id):
        self.selection_open(); self.require_pipeline()
        if not (self.root/'batch_plan.json').exists():
            raise RuntimeError('Run preflight first')
        name = dict(BACKBONES)[exp_id]
        tag = read_json(self.root/'backbone_tags.json')[name]['name']
        return run(self.config(exp_id,name,pretrained_tag=tag))

    def _baseline_config(self):
        selection = read_json(self.root/'backbone_selection.json')
        return Config(**selection['config'])

    def measure_backbones(self):
        """Measure every backbone on the current GPU; don't rank timings from different GPUs."""
        self.selection_open()
        rows=[]
        gpu=torch.cuda.get_device_name(0)
        for exp_id,name in BACKBONES:
            cfg = self.config(exp_id,name,pretrained_tag=read_json(self.root/'backbone_tags.json')[name]['name'])
            result = read_json(run_dir(cfg)/'result.json')
            latency_path = run_dir(cfg)/'latency.json'
            if latency_path.exists() and read_json(latency_path)['gpu']==gpu:
                latency=read_json(latency_path)
            else:
                model=load_trained(cfg)
                latency=pipeline_latency(model,batch_size=1)
                write_json(latency_path,latency)
                del model; gc.collect(); torch.cuda.empty_cache()
            rows.append({**result,'p95':latency['p95'],'p50':latency['p50'],'gpu':gpu,'config':asdict(cfg)})
        write_json(self.root/'backbone_comparison.json',rows)
        return rows

    def select_backbone(self):
        self.selection_open()
        rows=self.measure_backbones()
        best=sorted(rows,key=lambda r:(-round(r['macro_f1'],4),r['p95'],r['params_m']))[0]
        write_json(self.root/'backbone_selection.json',{'source_exp':best['exp_id'],'config':best['config'],
                   'rule':'macro-F1 rounded 4 digits descending, p95 ascending, params ascending','result':best})
        baseline={**best,'exp_id':'T00','source_exp':best['exp_id'],'axis':'baseline','change':'none (reused B seed0)'}
        write_json(self.root/'training_baseline.json',baseline)
        return best

    @record_failure
    def train_ablation(self, exp_id):
        self.selection_open(); self.require_pipeline()
        base=self._baseline_config()
        if exp_id=='T06':
            candidates=[]
            for loss_id in ('T04','T05'):
                result=read_json(run_dir(replace(base,exp_id=loss_id))/'result.json')
                candidates.append((result['macro_f1'],loss_id))
            baseline=read_json(self.root/'training_baseline.json')['macro_f1']
            score,loss_id=sorted(candidates,key=lambda p:(-p[0],p[1]))[0]
            if score <= baseline:
                loss_id='T04'
            change={'mix':'cutmix',**CHANGES[loss_id]}
            write_json(self.root/'combination_plan.json',{'source_loss':loss_id,'change':change,'baseline_f1':baseline})
        else:
            change=CHANGES[exp_id]
        cfg=replace(base,exp_id=exp_id,**change)
        result=run(cfg)
        write_json(run_dir(cfg)/'ablation.json',{'axis':{'T01':'A','T02':'A','T03':'B','T04':'C','T05':'C','T06':'B+C'}[exp_id],
                   'change':change,'delta':result['macro_f1']-read_json(self.root/'training_baseline.json')['macro_f1']})
        return result

    def select_recipe(self):
        self.selection_open()
        baseline=read_json(self.root/'training_baseline.json')
        rows=[{'exp_id':'T00','source_exp':baseline['source_exp'],'macro_f1':baseline['macro_f1'],'complexity':0}]
        base=self._baseline_config()
        for exp_id in ['T01','T02','T03','T04','T05','T06']:
            result=read_json(run_dir(replace(base,exp_id=exp_id))/'result.json')
            rows.append({**result,'source_exp':exp_id,'complexity':2 if exp_id=='T06' else 1})
        best=sorted(rows,key=lambda r:(-round(r['macro_f1'],4),r['complexity'],r['exp_id']))[0]
        cfg=Config(**read_json(Path(base.out_dir)/best['source_exp']/'seed0'/'config.json'))
        selection={'source_exp':best['source_exp'],'label':best['exp_id'],'config':asdict(cfg),'result':best,'candidates':rows}
        write_json(self.root/'recipe_selection.json',selection)
        return selection

    @record_failure
    def compare_inference(self):
        self.selection_open()
        cfg=Config(**read_json(self.root/'recipe_selection.json')['config'])
        _,val,_=load_split(cfg.labels_dir)
        model=load_trained(cfg)
        folder=self.root/'inference'; folder.mkdir(exist_ok=True)
        rows=[]
        gpu=torch.cuda.get_device_name(0)
        for exp_id,spec in METHODS.items():
            cache=folder/f'{exp_id}_val_logits.npz'
            if cache.exists():
                data=np.load(cache,allow_pickle=False)
                names,labels,z=data['filenames'].tolist(),data['y_true'],data['logits']
                if names!=val.Filename.tolist() or not np.array_equal(labels,val.Label.values):
                    raise ValueError('Inference cache order/labels changed')
            elif exp_id=='I04' and (folder/'I00_val_logits.npz').exists():
                data=np.load(folder/'I00_val_logits.npz',allow_pickle=False)
                names,labels,z=data['filenames'].tolist(),data['y_true'],data['logits']
                np.savez_compressed(cache,filenames=names,y_true=labels,logits=z)
            else:
                loader=make_loader(val,cfg.images_dir,inference_transforms(cfg.img_size,spec['method']=='fivecrop'),
                                   cfg.batch_size,False,num_workers=cfg.num_workers)
                names,labels,z=predict_pipeline(model,loader,'cuda',**spec,crop=cfg.img_size)
                np.savez_compressed(cache,filenames=names,y_true=labels,logits=z)
            temperature=fit_temperature(z,labels) if exp_id=='I04' else 1.
            probs=apply_temperature(z,temperature)
            save_predictions(folder/f'{exp_id}_seed0_val.csv',names,labels,probs)
            metrics=compute_metrics(labels,probs.argmax(1),probs)
            latencies=[]
            for batch in (1,32):
                path=folder/f'{exp_id}_batch{batch}_latency.json'
                if path.exists() and read_json(path)['gpu']==gpu:
                    latency=read_json(path)
                else:
                    latency=pipeline_latency(model,batch_size=batch,method=spec['method'],dtype='amp' if spec['amp'] else 'fp32',temperature=temperature)
                    write_json(path,latency)
                latencies.append(latency)
            rows.append({'exp_id':exp_id,'source_exp':cfg.exp_id,'method':spec['method'],'amp':spec['amp'],
                         'checkpoint':str((run_dir(cfg)/'best.pt').relative_to(self.root)),
                         'checkpoint_sha256':sha256(run_dir(cfg)/'best.pt'),
                         'temperature':temperature,'macro_f1':metrics['macro_f1'],'top1':metrics['top1'],
                         'ece':metrics['ece'],'nll':metrics['nll'],'K':{'single':1,'hflip':2,'fivecrop':5}[spec['method']],
                         'p50':latencies[0]['p50'],'p95':latencies[0]['p95'],'p99':latencies[0]['p99'],
                         'images_per_s':latencies[1]['images_per_s'],'gpu':gpu})
            print(exp_id,rows[-1],flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
        for row in rows:
            row['relative_cost']=row['p50']/rows[0]['p50']
        write_json(self.root/'inference_comparison.json',rows)
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,5))
        for row in rows:
            ax.scatter(row['p95'],row['macro_f1']); ax.annotate(row['exp_id'],(row['p95'],row['macro_f1']))
        ax.set_xlabel('p95 batch-1 latency (ms)'); ax.set_ylabel('Val macro-F1'); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(folder/'accuracy_latency.png',dpi=160); plt.close(fig)
        return rows

    def lock_final(self):
        path=self.root/'final_lock.json'
        if path.exists():
            return read_json(path)
        self.selection_open()
        rows=read_json(self.root/'inference_comparison.json')
        best=sorted([r for r in rows if r['exp_id']!='I04'],key=lambda r:(-round(r['macro_f1'],4),r['p95'],r['exp_id']))[0]
        data=np.load(self.root/'inference'/f'{best["exp_id"]}_val_logits.npz',allow_pickle=False)
        z,labels=data['logits'],data['y_true']
        temperature=fit_temperature(z,labels)
        raw=softmax(z); scaled=apply_temperature(z,temperature)
        before=compute_metrics(labels,raw.argmax(1),raw)['ece']
        after=compute_metrics(labels,scaled.argmax(1),scaled)['ece']
        lock={'baseline_config':asdict(self._baseline_config()),'final_config':read_json(self.root/'recipe_selection.json')['config'],
              'pipeline':{'method':best['method'],'amp':best['amp'],'calibrate':after<before},
              'calibration_policy':'If enabled by seed0 val, fit one T per trained seed on that seed val; never use test to fit or switch calibration',
              'selection_val_ece_before':before,'selection_val_ece_after':after,
              'seeds':[0,1,2],'epochs':10,'require_all_final_training':True,'code_hashes':code_hashes(),'notebook_url':self.notebook_url}
        write_json(path,lock)
        return lock

    @record_failure
    def final_train(self, exp_id, seed):
        lock=read_json(self.root/'final_lock.json')
        if exp_id not in ('T00','F01') or seed not in lock['seeds']:
            raise ValueError('final runs are T00/F01 seeds 0,1,2')
        if code_hashes()!=lock['code_hashes']:
            raise ValueError('Code changed after final lock; do not silently revise the study')
        # Train all six models before opening any test results.
        if (self.root/'test_started.json').exists() and not (self.root/'runs'/exp_id/f'seed{seed}'/'result.json').exists():
            raise RuntimeError('Finish all six final training runs before opening test')
        base=Config(**lock['baseline_config' if exp_id=='T00' else 'final_config'])
        cfg=replace(base,exp_id=exp_id,seed=seed,save_test_predictions=False)
        return run(cfg)

    @record_failure
    def final_test(self, exp_id, seed):
        lock=read_json(self.root/'final_lock.json')
        for group in ('T00','F01'):
            for k in lock['seeds']:
                if not (self.root/'runs'/group/f'seed{k}'/'result.json').exists():
                    raise RuntimeError(f'Complete all six final training runs first; missing {group} seed{k}')
        base=Config(**lock['baseline_config' if exp_id=='T00' else 'final_config'])
        cfg=replace(base,exp_id=exp_id,seed=seed,save_test_predictions=True)
        return final_predict(cfg)

    def export(self, student_id=None, student_name=None, notebook_url=None):
        from .reporting import export_results
        return export_results(self.root,self.data,student_id,student_name,notebook_url or self.notebook_url)

    def validate(self):
        from .validation import validate_completion
        return validate_completion(self.root,self.data,self.notebook_url)

    def verify_fresh_session(self):
        """Run in a restarted Colab session after setup; load models, never rerun test."""
        if read_json(self.root/'initial_session.json')['session_id']==SESSION_ID:
            raise RuntimeError('Restart the Colab session and rerun setup before this acceptance check')
        self.check_environment()
        lock=read_json(self.root/'final_lock.json')
        if lock['code_hashes']!=code_hashes(): raise ValueError('Code changed after final lock')
        checked=[]
        for exp in ('T00','F01'):
            for seed in range(3):
                folder=self.root/'runs'/exp/f'seed{seed}'
                if not (folder/'result.json').exists(): raise RuntimeError('Final training is incomplete')
                cfg=replace(Config(**read_json(folder/'config.json')),save_test_predictions=False)
                run(cfg)  # Completed run must pass compatibility and return without training.
                model=load_trained(cfg,device='cpu'); del model; gc.collect()
                checked.append(f'{exp}_seed{seed}')
        evidence={'session_id':SESSION_ID,'initial_session_id':read_json(self.root/'initial_session.json')['session_id'],
                  'code_hashes':code_hashes(),'environment':environment(),'checked':checked,'test_rerun':False}
        write_json(self.root/'fresh_session_validation.json',evidence)
        return evidence


def final_predict(cfg, model=None):
    root=Path(cfg.project_dir)
    lock=read_json(root/'final_lock.json')
    if cfg.exp_id not in ('T00','F01') or cfg.seed not in lock['seeds']:
        raise ValueError('Only locked final/baseline seeds are allowed on test')
    if lock.get('require_all_final_training',False):
        for group in ('T00','F01'):
            for seed in lock['seeds']:
                if not (root/'runs'/group/f'seed{seed}'/'result.json').exists():
                    raise RuntimeError('Complete all six final training runs before opening test')
    base=Config(**lock['baseline_config' if cfg.exp_id=='T00' else 'final_config'])
    expected=replace(base,exp_id=cfg.exp_id,seed=cfg.seed,save_test_predictions=True)
    if compatibility(cfg)!=compatibility(expected) or code_hashes()!=lock['code_hashes']:
        raise ValueError('Test configuration/code differs from locked study')
    folder=run_dir(cfg)
    if not (folder/'result.json').exists() and model is None:
        raise RuntimeError('Complete final training before test')
    pipeline={'method':'single','amp':False,'calibrate':False} if cfg.exp_id=='T00' else lock['pipeline']
    val_names,val_labels,val_logits=None,None,None
    if model is None:
        model=load_trained(cfg,cfg.device)
    _,val,test=load_split(cfg.labels_dir)
    cal_path=folder/'final_calibration.json'
    if cal_path.exists():
        calibration=read_json(cal_path)
        if calibration['lock_digest']!=lock_digest(lock):
            raise ValueError('Calibration belongs to a different lock')
        temperature=calibration['temperature']
    else:
        loader=make_loader(val,cfg.images_dir,inference_transforms(cfg.img_size,pipeline['method']=='fivecrop'),cfg.batch_size,False,num_workers=cfg.num_workers)
        val_names,val_labels,val_logits=predict_pipeline(model,loader,cfg.device,pipeline['method'],pipeline['amp'],cfg.img_size)
        temperature=fit_temperature(val_logits,val_labels) if pipeline['calibrate'] else 1.
        p=apply_temperature(val_logits,temperature)
        save_predictions(Path(cfg.pred_dir)/f'{cfg.exp_id}_seed{cfg.seed}_val.csv',val_names,val_labels,p)
        np.savez_compressed(folder/'final_val_logits.npz',filenames=val_names,y_true=val_labels,logits=val_logits)
        write_json(cal_path,{'temperature':temperature,'lock_digest':lock_digest(lock),'pipeline':pipeline,
                            'val_metrics':compute_metrics(val_labels,p.argmax(1),p)})
    output=Path(cfg.pred_dir)/f'{cfg.exp_id}_seed{cfg.seed}_test.csv'
    manifest_path=folder/'test_manifest.json'
    checkpoint_hash=sha256(folder/'best.pt')
    contract={'lock_digest':lock_digest(lock),'checkpoint_sha256':checkpoint_hash,'batch_size':cfg.batch_size,
              'test_csv_sha256':sha256(Path(cfg.labels_dir)/'test_subset0.csv'),'pipeline':pipeline,'temperature':temperature}
    if manifest_path.exists() and read_json(manifest_path)!=contract:
        raise ValueError('Test run contract changed; do not rerun or mix outputs')
    if output.exists() and (folder/'test_complete.json').exists():
        saved=read_json(folder/'test_complete.json')
        if saved['sha256']!=sha256(output):
            raise ValueError('Completed test predictions were modified')
        check_against_csv(read_pred(str(output)),str(Path(cfg.labels_dir)/'test_subset0.csv'))
        # A disconnect may happen after CSV completion but before latency measurement.
        if cfg.device.startswith('cuda'):
            for batch in (1,32):
                timing=folder/f'final_latency_batch{batch}.json'
                if not timing.exists():
                    write_json(timing,pipeline_latency(model,batch_size=batch,method=pipeline['method'],
                               dtype='amp' if pipeline['amp'] else 'fp32',temperature=temperature))
        print('Test already complete; inference skipped:',cfg.exp_id,cfg.seed)
        return str(output)
    write_json(manifest_path,contract)
    write_json(root/'test_started.json',{'locked':True,'message':'Selections are closed; never tune from test.'})
    parts=folder/'test_parts'; parts.mkdir(exist_ok=True)
    loader=make_loader(test,cfg.images_dir,inference_transforms(cfg.img_size,pipeline['method']=='fivecrop'),cfg.batch_size,False,num_workers=cfg.num_workers)
    model.eval()
    with torch.inference_mode():
        for index,(x,y,names) in enumerate(loader):
            cache=parts/f'batch{index:05d}.npz'
            if cache.exists():
                data=np.load(cache,allow_pickle=False)
                if data['filenames'].tolist()!=list(names) or not np.array_equal(data['y_true'],y.numpy()):
                    raise ValueError('Test batch order/labels changed')
                continue
            z=pipeline_logits(model,x.to(cfg.device),pipeline['method'],pipeline['amp'],cfg.img_size).cpu().numpy()
            tmp=cache.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,filenames=np.asarray(names),y_true=y.numpy(),logits=z)
            os.replace(tmp,cache)
    names,labels,logits=[],[],[]
    for index in range(len(loader)):
        data=np.load(parts/f'batch{index:05d}.npz',allow_pickle=False)
        names.extend(data['filenames'].tolist()); labels.extend(data['y_true'].tolist()); logits.append(data['logits'])
    if names!=test.Filename.tolist() or labels!=test.Label.tolist():
        raise ValueError('Incomplete or incorrectly ordered test inference')
    labels=np.asarray(labels); z=np.concatenate(logits)
    np.savez_compressed(folder/'test_logits.npz',filenames=names,y_true=labels,logits=z)
    p=apply_temperature(z,temperature)
    save_predictions(output,names,labels,p)
    check_against_csv(read_pred(str(output)),str(Path(cfg.labels_dir)/'test_subset0.csv'))
    if pipeline['calibrate']:
        save_predictions(Path(cfg.pred_dir)/f'{cfg.exp_id}uncal_seed{cfg.seed}_test.csv',names,labels,softmax(z))
    write_json(folder/'test_complete.json',{'sha256':sha256(output),'n':len(labels),'contract':contract})
    # Measure this exact final pipeline on the same active hardware, including calibration.
    if cfg.device.startswith('cuda'):
        for batch in (1,32):
            write_json(folder/f'final_latency_batch{batch}.json',pipeline_latency(model,batch_size=batch,
                       method=pipeline['method'],dtype='amp' if pipeline['amp'] else 'fp32',temperature=temperature))
    print('Completed test:',cfg.exp_id,cfg.seed,'n=',len(labels),flush=True)
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(output)
