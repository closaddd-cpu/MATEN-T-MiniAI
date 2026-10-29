# BY MATEN/T | MATEN/T MiniAI v1.1 | M/ License v1.0
"""One command: prepare data/tokenizer, run full causal pretraining and export.

No GPU work runs automatically when code is downloaded or published on GitHub.
The user must invoke this file explicitly on an adequately provisioned machine.
"""
from __future__ import annotations
from brand_guard import enforce_attribution
enforce_attribution()
import argparse
import json
from pathlib import Path
import subprocess
import sys
from build_datasets import FILES
from tools.verify_datasets import verify

ROOT=Path(__file__).resolve().parent


def execute(*args):
    print('BY MATEN/T |', ' '.join(map(str,args)),flush=True)
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)


def full_dataset(folder):
    path=Path(folder);manifest=path/'manifest.json'
    if not manifest.is_file():return False
    try:
        data=json.loads(manifest.read_text(encoding='utf-8'))
        domains=data['domains']
        if set(domains)!=set(FILES) or data.get('total_records')!=9_400_000:
            return False
        if any(domains[name]['rows']!=100_000 or not (path/(name+'.jsonl')).is_file() for name in FILES):return False
        return verify(path)
    except (OSError,KeyError,ValueError):return False


def prepare():
    regular=ROOT/'datasets'
    if full_dataset(regular):return regular
    # Never silently replace existing user data.
    target=ROOT/'datasets_full_100k' if any(regular.glob('*.jsonl')) else regular
    if not full_dataset(target):
        if any(target.glob('*.jsonl')):
            raise ValueError(f'Incomplete generated corpus already exists in {target}; please audit/remove it before rebuilding')
        execute(ROOT/'build_datasets.py','--out',target,'--count','100000')
        if not full_dataset(target):raise RuntimeError('Complete generated corpus did not pass integrity checks')
    return target


def main(argv=None):
    p=argparse.ArgumentParser(description='MATEN/T MiniAI v1.1 — full pretraining + safetensors')
    p.add_argument('--corpus',default='',help='Optional licensed .txt/.jsonl corpus; no synthetic data generation when set')
    p.add_argument('--tokenizer',default=str(ROOT/'tokenizer'/'tokenizer.model'))
    p.add_argument('--device',default='auto',choices=['auto','cuda','cpu','mps'])
    p.add_argument('--precision',default='auto',choices=['auto','fp16','bf16','fp32'])
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--sequence-length',type=int,default=512)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=8)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--output',default=str(ROOT/'pretrain_checkpoints'))
    p.add_argument('--lora-after',action='store_true',help='Fine-tune a LoRA adapter after full pretraining')
    a=p.parse_args(argv)
    if not 8<=a.sequence_length<=6144:raise ValueError('sequence-length must be 8..6144')
    if min(a.epochs,a.batch_size,a.grad_accum,a.save_every)<1:raise ValueError('Positive training settings are required')
    corpus=Path(a.corpus) if a.corpus else prepare()
    token=Path(a.tokenizer)
    if not token.is_file():
        from prepare_tokenizer import prepare as prepare_tok
        prepare_tok(corpus,token)
    from pretrain import parser as pretrain_parser, train
    out=Path(a.output)
    opts=pretrain_parser().parse_args(['--corpus',str(corpus),'--tokenizer',str(token),'--output',str(out),
            '--device',a.device,'--precision',a.precision,'--epochs',str(a.epochs),
            '--sequence-length',str(a.sequence_length),'--batch-size',str(a.batch_size),
            '--grad-accum',str(a.grad_accum),'--save-every',str(a.save_every)])
    last=out/'last.pt'
    if last.is_file():opts.resume=str(last)
    train(opts)
    if a.lora_after:
        from finetune_lora import parser as lora_parser,train as lora_train
        if not (corpus/'manifest.json').exists():
            raise ValueError('--lora-after requires instruction/response JSONL data; run LoRA separately for custom corpus')
        options=lora_parser().parse_args(['--base',str(last),'--data',str(corpus),'--tokenizer',str(token),'--device',a.device,'--precision',a.precision])
        lora_train(options)
    print('BY MATEN/T | Pretraining workflow complete. Export files:',out/'release',flush=True)

if __name__=='__main__':main()
