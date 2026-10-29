# BY MATEN/T | MATEN/T MiniAI v1.1 | M/ License v1.0
"""LoRA supervised fine-tuning AFTER causal pretraining, for JSONL dialogues."""
from __future__ import annotations
from brand_guard import enforce_attribution
enforce_attribution()
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from model import ModelConfig,MiniAI
from lora import inject_lora,adapter_state
from pretrain import device_for,precision_for,autocast,scaler_for,source_signature
from miniai import file_rows,example,collate

ROOT=Path(__file__).resolve().parent


def dialogue_batches(paths,tokenizer,seq_len,batch_size,skip=0,limit=0):
    pending=[];batch_no=0
    for row in file_rows(paths,'train',max_rows=limit):
        if not isinstance(row,dict) or not isinstance(row.get('instruction'),str) or not isinstance(row.get('response'),str):
            raise ValueError('LoRA training expects instruction/response JSONL rows')
        pending.append(example(row,tokenizer,seq_len))
        if len(pending)==batch_size:
            if batch_no>=skip:yield collate(pending,tokenizer.pad_id())
            pending=[];batch_no+=1
    if pending and batch_no>=skip:yield collate(pending,tokenizer.pad_id())


def train(args):
    if min(args.epochs,args.rank,args.batch_size,args.grad_accum,args.save_every)>0:pass
    else:raise ValueError('Epochs, rank, batch, accumulation, save interval must be positive')
    device=device_for(args.device);precision=precision_for(device,args.precision)
    ck=torch.load(args.base,map_location='cpu',weights_only=True)
    if ck.get('training_mode')!='causal_pretraining':
        raise ValueError('LoRA requires a causal_pretraining checkpoint, not random weights or SFT checkpoint.')
    cfg=ModelConfig(**ck['config'])
    if args.sequence_length>cfg.context_length or args.sequence_length<16:raise ValueError('Invalid sequence length')
    import sentencepiece as spm
    tok_path=Path(args.tokenizer)
    tok=spm.SentencePieceProcessor(model_file=str(tok_path))
    model=MiniAI(cfg);model.load_state_dict(ck['model'])
    inject_lora(model,args.rank,args.alpha,args.dropout)
    model.to(device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=0.0)
    scaler=scaler_for(precision)
    paths=sorted(p for p in Path(args.data).rglob('*.jsonl') if p.is_file())
    if not paths:raise FileNotFoundError('No JSONL fine-tuning dataset files')
    source_sig=source_signature(paths)
    tok_hash=hashlib.sha256(tok_path.read_bytes()).hexdigest()
    base_hash=hashlib.sha256()
    with Path(args.base).open('rb') as fp:
        while chunk:=fp.read(4<<20):base_hash.update(chunk)
    base_hash=base_hash.hexdigest()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    conf={'created_by':'MATEN/T','version':'1.1','rank':args.rank,'alpha':args.alpha,
          'dropout':args.dropout,'targets':['qkv','proj'],'base_checkpoint_sha256':base_hash,
          'tokenizer_sha256':tok_hash,'data_signature':source_sig,'base_config':asdict(cfg),
          'sequence_length':args.sequence_length,'batch_size':args.batch_size,
          'grad_accum':args.grad_accum,'lr':args.lr,'precision':precision,'max_records':args.max_records}
    position=0;start_epoch=0;step=0;total_tokens=0
    resume=output/'trainer_last.pt'
    if args.resume:
        if not resume.is_file():raise FileNotFoundError(resume)
        previous=torch.load(resume,map_location='cpu',weights_only=True)
        if previous['settings']!=conf:raise ValueError('LoRA resume config, data or base checkpoint differs')
        own=model.state_dict()
        for key,v in previous['adapter'].items():own[key].copy_(v)
        optimizer.load_state_dict(previous['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to(device)
        scaler.load_state_dict(previous['scaler'])
        start_epoch=previous['epoch'];position=previous['micro_cursor'];step=previous['step'];total_tokens=previous['tokens_seen']

    def save(epoch,cursor):
        tensors=adapter_state(model)
        temp=output/'adapter.tmp.safetensors'
        save_file(tensors,str(temp),metadata={'creator':'MATEN/T','adapter_type':'LoRA'})
        os.replace(temp,output/'adapter_model.safetensors')
        (output/'adapter_config.json').write_text(json.dumps(conf,indent=2)+'\n',encoding='utf-8')
        package={'settings':conf,'adapter':tensors,'optimizer':optimizer.state_dict(),
                 'scaler':scaler.state_dict(),'epoch':epoch,'micro_cursor':cursor,'step':step,
                 'tokens_seen':total_tokens}
        tmp=output/'trainer_last.tmp'
        torch.save(package,tmp);os.replace(tmp,resume)
        print(f'BY MATEN/T | LoRA checkpoint step={step} tokens={total_tokens:,}',flush=True)

    print(f'BY MATEN/T | LoRA trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,} / total={sum(p.numel() for p in model.parameters()):,}',flush=True)
    for epoch in range(start_epoch,args.epochs):
        model.train();optimizer.zero_grad(set_to_none=True);acc=0;cursor=position if epoch==start_epoch else 0
        for x,y in dialogue_batches(paths,tok,args.sequence_length,args.batch_size,skip=cursor,limit=args.max_records):
            if not bool((y!=-100).any()):continue
            total_tokens+=int((y!=-100).sum().item())
            x=x.to(device);y=y.to(device)
            with autocast(device,precision):_,loss=model(x,y)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite LoRA loss')
            scaler.scale(loss/args.grad_accum).backward();acc+=1;cursor+=1
            if acc<args.grad_accum:continue
            scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0)
            old=scaler.get_scale();scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);acc=0
            if not (precision=='fp16' and scaler.get_scale()<old):step+=1
            if step and step%args.save_every==0:save(epoch,cursor)
            if args.max_steps and step>=args.max_steps:
                save(epoch,cursor);return output
        if acc:
            if acc!=args.grad_accum:
                for p in model.parameters():
                    if p.grad is not None:p.grad.mul_(args.grad_accum/acc)
            scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0)
            scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);step+=1
        save(epoch+1,0);position=0
    from export_model import export_checkpoint
    export_checkpoint(args.base,output/'release_merged',tok_path,precision='fp16',adapter_dir=output)
    return output


def parser():
    p=argparse.ArgumentParser(description='BY MATEN/T | train a LoRA adapter after pretraining')
    p.add_argument('--base',required=True,help='Trusted pretrained checkpoint: pretrain_checkpoints/last.pt')
    p.add_argument('--data',default=str(ROOT/'datasets'))
    p.add_argument('--tokenizer',default=str(ROOT/'tokenizer'/'tokenizer.model'))
    p.add_argument('--output',default=str(ROOT/'lora_output'))
    p.add_argument('--device',default='auto',choices=['auto','cuda','cpu','mps'])
    p.add_argument('--precision',default='auto',choices=['auto','fp16','bf16','fp32'])
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--alpha',type=float,default=16)
    p.add_argument('--dropout',type=float,default=0.05)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=8)
    p.add_argument('--sequence-length',type=int,default=512)
    p.add_argument('--save-every',type=int,default=200)
    p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--max-records',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    return p

if __name__=='__main__':train(parser().parse_args())
