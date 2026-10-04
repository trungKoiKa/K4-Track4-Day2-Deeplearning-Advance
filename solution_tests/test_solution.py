import copy
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eval import read_pred, save_predictions
from lab_solution import dataset, inference, losses, model, train
from lab_solution.common import code_hashes, write_json
from lab_solution.experiments import Study, final_predict


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,4,1)
        self.bn = nn.BatchNorm2d(4)
        self.head = nn.Linear(4,9)
        self.calls = 0
        self.pretrained_cfg = {'tag':'test','mean':(.485,.456,.406),'std':(.229,.224,.225)}

    def get_classifier(self):
        return self.head

    def forward(self,x):
        self.calls += 1
        return self.head(self.bn(self.conv(x)).mean((2,3)))


class SolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_loss_equivalence_and_temperature(self):
        torch.manual_seed(0)
        z = torch.randn(20,9)
        y = torch.randint(9,(20,))
        ce = nn.functional.cross_entropy(z,y)
        torch.testing.assert_close(losses.FocalLoss(0)(z,y),ce,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(losses.LabelSmoothingCE(0)(z,y),ce)
        t = inference.fit_temperature(z.numpy()*4,y.numpy())
        p = inference.apply_temperature(z.numpy()*4,t)
        self.assertTrue(np.allclose(p.sum(1),1))
        self.assertTrue(np.array_equal(p.argmax(1),z.numpy().argmax(1)))
        nll = -np.log(p[np.arange(20),y.numpy()]).mean()
        self.assertLessEqual(nll,float(nn.functional.cross_entropy(z*4,y))+1e-6)
        with self.assertRaises(ValueError): inference.apply_temperature(z.numpy(),0)

    def test_cutmix_area_and_label_pairing(self):
        torch.manual_seed(4); np.random.seed(4)
        # A two-example image batch makes each changed pixel reveal the actual patch area.
        x=torch.stack([torch.zeros(3,9,11),torch.ones(3,9,11)])
        y=torch.tensor([2,7])
        mixed,(a,b,lam)=losses.mix_batch(x,y)
        changed=float((mixed[0]!=x[0]).float().mean())
        if int(b[0])!=int(a[0]): self.assertAlmostEqual(changed,1-lam,places=6)
        self.assertEqual(sorted(b.tolist()),sorted(y.tolist()))
        criterion=nn.CrossEntropyLoss(); z=torch.randn(2,9)
        torch.testing.assert_close(losses.mixed_loss(criterion,z,(a,b,lam)),lam*criterion(z,a)+(1-lam)*criterion(z,b))

    def test_decay_freeze_and_bn_fusion(self):
        net=Tiny()
        groups=model.param_groups(net,1e-4,1e-3,.05)
        assignment={id(p):g for g in groups for p in g['params']}
        self.assertEqual(len(assignment),len(list(net.parameters())))
        self.assertEqual(assignment[id(net.head.bias)]['weight_decay'],0)
        self.assertEqual(assignment[id(net.bn.weight)]['weight_decay'],0)
        self.assertEqual(assignment[id(net.head.weight)]['lr'],1e-3)
        model.freeze_backbone(net); net.train(); model.frozen_train_mode(net)
        before=net.bn.running_mean.clone()
        net(torch.randn(4,3,8,8)).sum().backward()
        torch.testing.assert_close(before,net.bn.running_mean)
        self.assertIsNone(net.conv.weight.grad)
        self.assertIsNotNone(net.head.weight.grad)
        seq=nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.BatchNorm2d(4),nn.ReLU()).eval()
        fused=inference.fuse_conv_bn(seq)
        x=torch.randn(3,3,8,8)
        torch.testing.assert_close(seq(x),fused(x),atol=1e-5,rtol=1e-5)

    def test_multicrop_distinct_views_and_predictions_contract(self):
        x=torch.arange(256*256).reshape(1,1,256,256).float()
        crops=inference.views_multicrop(x,224)
        self.assertEqual(len(crops),5)
        self.assertEqual(len({float(c[0,0,0,0]) for c in crops}),5)
        torch.testing.assert_close(inference.view_hflip(inference.view_hflip(x)),x)
        p=inference.aggregate_views([np.zeros((9,9)),np.ones((9,9))],'logit')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'F01_seed2_test.csv'
            save_predictions(path,[f'{i}.jpg' for i in range(9)],np.arange(9),p)
            self.assertEqual(read_pred(str(path)).seed,2)

    def test_dataset_optional_species_and_stable_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            df=pd.DataFrame({'Filename':['b.jpg','a.jpg'],'Label':[3,4]})
            for name in df.Filename: Image.new('RGB',(256,256),(50,80,10)).save(root/name)
            for split in ('train','val','test'): df.to_csv(root/f'{split}_subset0.csv',index=False)
            loaded=dataset.load_split(root)[0]
            loader=dataset.make_loader(loaded,root,dataset.build_transforms(False),1,False,num_workers=0)
            batches=list(loader)
            self.assertEqual([b[2][0] for b in batches],['b.jpg','a.jpg'])
            self.assertEqual(tuple(batches[0][0].shape),(1,3,224,224))

    def test_gradient_accumulation_partial_window_matches_reference(self):
        torch.manual_seed(3)
        net=nn.Linear(2,9); ref=copy.deepcopy(net)
        x=torch.randn(6,2); y=torch.tensor([0,1,2,3,4,5])
        loader=[(x[i:i+2],y[i:i+2],None) for i in range(0,6,2)]
        cfg=train.Config(grad_accum=2,amp=False,device='cpu')
        opt=torch.optim.SGD(net.parameters(),lr=.1)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        train.train_one_epoch(net,loader,nn.CrossEntropyLoss(),opt,scheduler,scaler,cfg,'cpu')
        refopt=torch.optim.SGD(ref.parameters(),lr=.1)
        for start,end in [(0,4),(4,6)]:
            refopt.zero_grad(); nn.functional.cross_entropy(ref(x[start:end]),y[start:end]).backward(); refopt.step()
        for a,b in zip(net.parameters(),ref.parameters()): torch.testing.assert_close(a,b)

    def test_training_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); labels=root/'labels'; labels.mkdir()
            df=pd.DataFrame({'Filename':[f'{i}.jpg' for i in range(18)],'Label':np.arange(18)%9})
            for name in ('labels.csv','train_subset0.csv','val_subset0.csv','test_subset0.csv'): df.to_csv(labels/name,index=False)
            torch.manual_seed(99); x=torch.randn(18,3,8,8); y=torch.arange(18)%9
            class Batches:
                def __init__(self): self.generator=torch.Generator()
                def __len__(self): return 3
                def __iter__(self):
                    for start in (0,6,12): yield x[start:start+6],y[start:start+6],df.Filename.iloc[start:start+6].tolist()
            common=dict(labels_dir=str(labels),images_dir=str(root),out_dir=str(root/'runs'),pred_dir=str(root/'pred'),
                        curves_dir=str(root/'curves'),project_dir=str(root),device='cpu',amp=False,epochs=2,batch_size=6,grad_accum=2,profile=False,num_workers=0)
            patches=[patch('lab_solution.train.load_split',return_value=(df,df,df)),
                     patch('lab_solution.train.check_split',return_value={'test_fixture':True}),
                     patch('lab_solution.train.make_loader',side_effect=lambda *a,**k:Batches()),
                     patch('lab_solution.train.build_model',side_effect=lambda *a,**k:Tiny())]
            for p in patches: p.start()
            try:
                full=train.Config(exp_id='full',**common); resumed=train.Config(exp_id='resume',**common)
                train.run(full)
                with patch('lab_solution.train.plot_curves',side_effect=RuntimeError('simulated interruption')):
                    with self.assertRaisesRegex(RuntimeError,'interruption'): train.run(resumed)
                self.assertEqual(torch.load(train.run_dir(resumed)/'last.pt',weights_only=False)['epoch'],0)
                train.run(resumed)
                a=torch.load(train.run_dir(full)/'last.pt',weights_only=False)
                b=torch.load(train.run_dir(resumed)/'last.pt',weights_only=False)
                best=torch.load(train.run_dir(full)/'best.pt',weights_only=False)
                for state in (a,b,best):
                    self.assertTrue({'training_model','evaluation_model','model','optimizer','scheduler','scaler','ema','rng','metadata','history','epoch'}.issubset(state))
                for key in a['model']: torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)
                self.assertEqual(a['scheduler'],b['scheduler'])
                with self.assertRaisesRegex(ValueError,'changed'): train.run(replace(resumed,lr_head=.2))
            finally:
                for p in reversed(patches): p.stop()

    def test_final_test_cache_prevents_second_inference_and_rejects_changed_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); labels=root/'labels'; labels.mkdir(); images=root/'images'; images.mkdir()
            df=pd.DataFrame({'Filename':[f'{i}.jpg' for i in range(9)],'Label':range(9)})
            for name in df.Filename: Image.new('RGB',(32,32),(90,80,70)).save(images/name)
            for split in ('train','val','test'): df.to_csv(labels/f'{split}_subset0.csv',index=False)
            cfg=train.Config(exp_id='F01',seed=0,project_dir=str(root),images_dir=str(images),labels_dir=str(labels),
                            out_dir=str(root/'runs'),pred_dir=str(root/'pred'),device='cpu',batch_size=3,num_workers=0,img_size=16,profile=False)
            lock={'baseline_config':asdict(cfg),'final_config':asdict(cfg),'pipeline':{'method':'single','amp':False,'calibrate':True},'seeds':[0,1,2],'code_hashes':code_hashes()}
            write_json(root/'final_lock.json',lock)
            net=Tiny(); train.run_dir(cfg).mkdir(parents=True)
            torch.save({'model':net.state_dict()},train.run_dir(cfg)/'best.pt')
            with patch('lab_solution.experiments.load_split',return_value=(df,df,df)):
                final_predict(cfg,net)
                first_calls=net.calls
                final_predict(cfg,net)
                self.assertEqual(net.calls,first_calls)
                self.assertEqual(len(read_pred(str(root/'pred'/'F01_seed0_test.csv')).y_true),9)
                with self.assertRaisesRegex(ValueError,'locked'): final_predict(replace(cfg,loss='focal'),net)
            with self.assertRaisesRegex(RuntimeError,'locked'): Study(root,root).selection_open()


if __name__=='__main__': unittest.main()
