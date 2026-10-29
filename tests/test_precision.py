# BY MATEN/T | M/ License v1.0
import sys
from pathlib import Path
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from miniai import resolve_precision, cast_for, make_scaler, learning_rate_at
from model import ModelConfig,MiniAI

class PrecisionTests(unittest.TestCase):
    def test_cpu_falls_back_to_fp32(self):
        self.assertEqual(resolve_precision(torch.device('cpu'),'auto'),'fp32')
        with cast_for(torch.device('cpu'),'auto'):
            x=torch.ones(2)
        self.assertEqual(x.dtype,torch.float32)
    def test_explicit_fp16_requires_cuda(self):
        with self.assertRaisesRegex(ValueError,'requires a CUDA device'):
            resolve_precision(torch.device('cpu'),'fp16')
    def test_scaler_disabled_for_cpu(self):
        self.assertFalse(make_scaler(torch.device('cpu'),'fp32').is_enabled())
    def test_warmup_and_cosine_decay(self):
        rate=lambda step:learning_rate_at(step,0.001,10,100,0.1)
        self.assertAlmostEqual(rate(0),0.0001)
        self.assertAlmostEqual(rate(9),0.001)
        self.assertAlmostEqual(rate(100),0.0001)
        self.assertGreater(rate(20),rate(50))
    def test_small_model_training_step(self):
        cfg=ModelConfig(vocab_size=256,context_length=32,n_layer=1,n_head=4,d_model=32,d_ff=64,dropout=0)
        m=MiniAI(cfg)
        x=torch.randint(0,256,(2,8))
        opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
        with cast_for(torch.device('cpu'),'auto'):
            _,loss=m(x,x)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward();opt.step()
        self.assertTrue(torch.isfinite(loss).item())
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available in this environment')
    def test_actual_fp16_cuda_step(self):
        dev=torch.device('cuda')
        m=MiniAI(ModelConfig(vocab_size=256,context_length=32,n_layer=1,n_head=4,d_model=32,d_ff=64,dropout=0)).to(dev)
        x=torch.randint(0,256,(2,8),device=dev)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
        scaler=make_scaler(dev,'fp16')
        self.assertTrue(scaler.is_enabled())
        with cast_for(dev,'fp16'):
            _,loss=m(x,x)
        self.assertTrue(torch.isfinite(loss).item())
        scaler.scale(loss).backward();scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        scaler.step(opt);scaler.update()

if __name__=='__main__':unittest.main()
