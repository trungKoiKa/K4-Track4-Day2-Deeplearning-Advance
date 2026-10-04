"""Acceptance uses temporary synthetic evidence, never lab results."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import eval as ev
from lab_solution.common import write_json, code_hashes, sha256
from lab_solution.validation import validate_completion


class ValidationTests(unittest.TestCase):
    def test_complete_fixture_and_missing_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'study'; data=Path(temp)/'data'; root.mkdir(); (data/'labels').mkdir(parents=True)
            names=[f'{i}.jpg' for i in range(9)]; y=np.arange(9)
            df=pd.DataFrame({'Filename':names,'Label':y})
            for split in ('test','val'):df.to_csv(data/'labels'/f'{split}_subset0.csv',index=False)
            runs=[(f'B{i:02d}',0) for i in range(1,6)]+[(f'T{i:02d}',0) for i in range(1,7)]+[(e,s) for e in ('T00','F01') for s in range(3)]
            def file(path):
                p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic fixture');return p
            def latency(path,batch):
                write_json(root/path,{'batch':batch,'warmup':10,'n':100,'gpu':'synthetic GPU'})
            for e,s in runs:
                folder=Path('runs')/e/f'seed{s}'
                write_json(root/folder/'config.json',{'epochs':10,'seed':s,'exp_id':e,'fold':0})
                write_json(root/folder/'result.json',{'epochs':10,'seed':s,'exp_id':e})
                write_json(root/folder/'metadata.json',{'code_hashes':code_hashes()})
                for name in ('best.pt','last.pt','environment.json','pretrained.json','profile.json','val_logits.npz'):file(folder/name)
                pd.DataFrame({'epoch':range(1,11),'train_loss':1.,'val_loss':1.,'macro_f1':.5,'top1':.5}).to_csv(root/folder/'history.csv',index=False)
                file(Path('curves')/f'{e}_seed{s}.png')
                ev.save_predictions(root/'predictions'/f'{e}_seed{s}_val.csv',names,y,np.eye(9))
                if e in ('T00','F01'):
                    pred=root/'predictions'/f'{e}_seed{s}_test.csv'
                    ev.save_predictions(pred,names,y,np.eye(9))
                    write_json(root/folder/'test_complete.json',{'sha256':sha256(pred)})
                    for batch in (1,32):latency(folder/f'final_latency_batch{batch}.json',batch)
            for i in range(5):
                for batch in (1,32):latency(Path('inference')/f'I{i:02d}_batch{batch}_latency.json',batch)
                file(Path('inference')/f'I{i:02d}_val_logits.npz')
            write_json(root/'pipeline_checks.json',{'passed':True,'overfit_loss':.02})
            write_json(root/'eda'/'split_checks.json',{'n':{'train':10501,'val':3501,'test':3507},'union':17509,'overlap':{'train_val':0},'missing':0})
            for name in ('eda/class_counts.csv','eda/class_distribution.png','eda/train_samples_3_per_class.png','eda/image_metadata.json',
                         'pipeline/overfit_batch.png','pipeline/augmentation_cutmix.png','runtime_versions.json','environment.lock.txt',
                         'backbone_comparison.json','training_baseline.json','recipe_selection.json','inference_comparison.json'):
                file(Path(name))
            write_json(root/'final_lock.json',{'code_hashes':code_hashes(),'pipeline':{'calibrate':False}})
            write_json(root/'fresh_session_validation.json',{'session_id':'new','initial_session_id':'old','code_hashes':code_hashes(),'checked':list(range(6)),'test_rerun':False})
            (root/'analysis_notes.md').write_text('synthetic reviewed notes '*20)
            url='https://colab.research.google.com/drive/synthetic-test'
            result=validate_completion(root,data,url,require_exports=False)
            self.assertTrue(result['ready_to_submit'],result['errors'])
            (root/'curves'/'B01_seed0.png').unlink()
            (root/'runs'/'F01'/'seed2'/'final_latency_batch32.json').unlink()
            result=validate_completion(root,data,url,require_exports=False)
            self.assertFalse(result['ready_to_submit'])
            self.assertTrue(any('B01_seed0.png' in e for e in result['errors']))
            self.assertTrue(any('final_latency_batch32.json' in e for e in result['errors']))
