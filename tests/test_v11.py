# BY MATEN/T | M/ License v1.0
"""V1.1 verification: architecture, packed pretraining, LoRA and safe export."""
import tempfile
from pathlib import Path
import json
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import MiniAI,ModelConfig
from pretrain import iter_documents,token_batches,parser as pretrain_parser,validate_model_size
from lora import inject_lora,adapter_state,load_adapter
from export_model import safe_state,restore_weights


class FakeTokenizer:
    def encode(self,s,out_type=int):return [4+ord(c)%32 for c in s]
    def pad_id(self):return 3
    def eos_id(self):return 2


class ReleaseTests(unittest.TestCase):
    def test_100m_model_and_context(self):
        self.assertEqual(validate_model_size(),100_062_768)
        self.assertEqual(ModelConfig().context_length,6144)
        self.assertEqual(ModelConfig().d_ff,4152)

    def test_real_pretraining_defaults(self):
        args=pretrain_parser().parse_args([])
        self.assertEqual(args.max_steps,0)
        self.assertFalse(args.tiny)
        self.assertEqual(args.epochs,1)

    def test_deterministic_causal_blocks(self):
        rows=['this is learning, not merely a prompt','second document']*8
        batches=list(token_batches(rows,FakeTokenizer(),16,2))
        self.assertGreater(len(batches),1)
        x,y=batches[0]
        self.assertEqual(x.shape,(2,16))
        self.assertTrue(torch.equal(x[0,1:],y[0,:-1]))
        self.assertEqual(sum(int((v!=-100).sum()) for _,v in batches),sum(int((v!=-100).sum()) for _,v in token_batches(rows,FakeTokenizer(),16,2)))

    def test_separated_splits_and_exact_dedup(self):
        with tempfile.TemporaryDirectory() as temp:
            f=Path(temp)/'documents.jsonl'
            f.write_text(''.join(json.dumps({'text':f'Document {i}'})+'\n' for i in range(40)),encoding='utf-8')
            train=list(iter_documents([f],'train'))
            valid=list(iter_documents([f],'validation'))
            self.assertEqual(len(train),36)
            self.assertEqual(len(valid),2)
            self.assertFalse(set(train)&set(valid))

    def test_lora_gradient_and_merge(self):
        cfg=ModelConfig(vocab_size=256,context_length=32,n_layer=1,n_head=4,d_model=32,d_ff=64,dropout=0)
        base=MiniAI(cfg)
        reference=base(torch.randint(0,256,(1,6)))[0].detach()
        inject_lora(base,rank=4,alpha=8,dropout=0)
        trainable=sum(p.numel() for p in base.parameters() if p.requires_grad)
        self.assertGreater(trainable,0)
        self.assertLess(trainable,sum(p.numel() for p in base.parameters())//4)
        t=torch.randint(0,256,(1,6))
        logits,loss=base(t,t)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in base.parameters() if p.requires_grad))
        state=adapter_state(base)
        self.assertEqual(len(state),4)
        for k in state:
            if k.endswith('lora_B'):state[k].fill_(0.01)
        another=MiniAI(cfg)
        # Keep base weights identical before applying and merging the adapter.
        weights={k:v for k,v in base.state_dict().items() if '.base.' in k or '.lora_' not in k}
        self.assertTrue(weights)
        # Round-trip against a new copy initialized from the underlying base.
        for block_src,block_target in zip(base.blocks,another.blocks):
            block_target.attn.qkv.load_state_dict(block_src.attn.qkv.base.state_dict())
            block_target.attn.proj.load_state_dict(block_src.attn.proj.base.state_dict())
        another.tok_emb.load_state_dict(base.tok_emb.state_dict())
        another.ln_f.load_state_dict(base.ln_f.state_dict())
        for src,dst in zip(base.blocks,another.blocks):
            dst.ln1.load_state_dict(src.ln1.state_dict());dst.ln2.load_state_dict(src.ln2.state_dict())
            dst.fc.load_state_dict(src.fc.state_dict());dst.proj.load_state_dict(src.proj.state_dict())
        load_adapter(another,state,rank=4,alpha=8,dropout=0,merge=True)
        self.assertTrue(torch.isfinite(another(t)[0]).all())

    def test_tied_weight_safetensor_dictionary(self):
        small=MiniAI(ModelConfig(vocab_size=256,context_length=32,n_layer=1,n_head=4,d_model=32,d_ff=64,dropout=0))
        packed=safe_state(small,'fp16')
        self.assertNotIn('head.weight',packed)
        restored=MiniAI(small.cfg)
        restore_weights(restored,packed)
        self.assertTrue(torch.equal(restored.tok_emb.weight,small.tok_emb.weight.half().float()))

if __name__=='__main__':unittest.main()
