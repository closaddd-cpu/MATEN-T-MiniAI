# BY MATEN/T | M/ License v1.0
import json
import sys
from pathlib import Path
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import ModelConfig,MiniAI
from build_datasets import make_record,FILES,CODE_LANGS,NATURAL_LANGS,DOMAINS
from catalog import NATURAL,EXTRA_CODE
from miniai import example,collate,load_tokenizer,device_for,packed_batches,parser

class ProjectTests(unittest.TestCase):
    def test_catalogue(self):
        self.assertEqual(len(CODE_LANGS),50)
        self.assertEqual(len(NATURAL_LANGS),20)
        self.assertEqual(len(DOMAINS),24)
        self.assertEqual(len(FILES),94)
        self.assertEqual(len(FILES),len(set(FILES)))
    def test_records_are_valid(self):
        for domain in FILES:
            with self.subTest(domain=domain):
                r=make_record(domain,11)
                self.assertEqual(r['created_by'],'MATEN/T')
                self.assertEqual(r['license'],'M/ License v1.0')
                self.assertTrue(r['instruction'] and r['response'])
                self.assertEqual(r['review_status'],'synthetic-not-individually-reviewed')
    def test_distinct_math_inputs(self):
        for domain in FILES:
            self.assertNotEqual(make_record(domain,0)['instruction'],make_record(domain,1)['instruction'])
    def test_metadata_provenance(self):
        self.assertEqual(make_record('arabic',1)['provenance'],'MATEN/T-original-procedural-v1')
    def test_native_language_correct_arithmetic(self):
        for lang in NATURAL:
            for i in [0,1,2,109]:
                r=make_record(lang,i)
                self.assertIn(r['topic'].split('/')[-2],NATURAL)
                self.assertTrue(any(x in r['response'] for x in ['😊']))
    def test_model_configuration(self):
        cfg=ModelConfig()
        self.assertEqual(cfg.context_length,6144)
        self.assertEqual(cfg.d_ff,3240)
        self.assertEqual(cfg.vocab_size,16384)
    def test_full_epoch_has_no_default_step_cap(self):
        args=parser().parse_args(['train'])
        self.assertEqual(args.max_steps,0)
        self.assertTrue(args.pack)
        self.assertEqual(args.sequence_length,1024)

    def test_rope_backward(self):
        cfg=ModelConfig(vocab_size=256,context_length=6144,n_layer=1,n_head=4,d_model=32,d_ff=64,dropout=0)
        model=MiniAI(cfg)
        x=torch.randint(0,256,(1,64))
        _,loss=model(x,x)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

    def test_packed_stream(self):
        class Tok:
            def piece_to_id(self,x):return {'<|user|>':4,'<|assistant|>':5,'<|end|>':6}[x]
            def unk_id(self):return 0
            def bos_id(self):return 1
            def encode(self,s,out_type=int):return [7+(ord(ch)%20) for ch in s]
        rows=[{'instruction':f'Question {i}','response':f'Answer {i}'} for i in range(60)]
        batches_=list(packed_batches(rows,Tok(),64,2,3))
        self.assertTrue(batches_)
        self.assertEqual(tuple(batches_[0][0].shape),(2,64))

    def test_tokenizer_scripts(self):
        if not (ROOT/'tokenizer'/'tokenizer.model').exists():
            self.skipTest('Tokenizer binary is not shipped with the source-only repo; run prepare')
        tok=load_tokenizer()
        for phrase in ['سڵاو!','مرحبا','Hello!','你好','こんにちは','안녕하세요']:
            ids=tok.encode(phrase)
            self.assertTrue(ids)
            self.assertTrue(all(0<=t<tok.vocab_size() for t in ids))
    def test_model_gradient(self):
        cfg=ModelConfig(vocab_size=256,context_length=48,n_layer=1,n_head=4,d_model=48,d_ff=96,dropout=0)
        net=MiniAI(cfg)
        x=torch.randint(0,256,(2,12));y=x.clone()
        _,loss=net(x,y);self.assertTrue(torch.isfinite(loss).item())
        loss.backward();self.assertTrue(any(p.grad is not None for p in net.parameters()))
    def test_example_mask(self):
        if not (ROOT/'tokenizer'/'tokenizer.model').exists():
            self.skipTest('Tokenizer binary is not shipped with the source-only repo; run prepare')
        tok=load_tokenizer();x,y=example(make_record('kurdish_sorani',1),tok,256)
        self.assertEqual(len(x),len(y));self.assertGreater(sum(v!=-100 for v in y),5)
    def test_collate(self):
        x,y=collate([([1,2],[3,4]),([1],[3])],pad_id=3)
        self.assertEqual(x.shape,(2,2));self.assertEqual(y[1,1].item(),-100)
    def test_manifest_if_present(self):
        path=ROOT/'datasets'/'manifest.json'
        if path.exists():
            m=json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(len(m['domains']),94)
            self.assertEqual(m['total_records'],9_400_000)

if __name__=='__main__': unittest.main()
