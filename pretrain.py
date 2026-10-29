# BY MATEN/T | MATEN/T MiniAI v1.1 | M/ License v1.0
"""Streaming next-token pretraining from random weights; no default step cap.

Inputs: folders/files with .txt or .jsonl. JSONL supports {"text": ...} or
{"instruction": ..., "response": ...}. This is language-model pretraining:
ALL content tokens are targets, not just assistant response tokens.
"""
from __future__ import annotations
from brand_guard import enforce_attribution
enforce_attribution()
import argparse
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import torch
from model import ModelConfig, MiniAI

ROOT = Path(__file__).resolve().parent


def device_for(choice: str):
    if choice == 'auto':
        if torch.cuda.is_available(): return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps')
        return torch.device('cpu')
    if choice == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: install matching PyTorch and driver or use --device cpu.')
    if choice == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        raise RuntimeError('MPS unavailable.')
    return torch.device(choice)


def precision_for(device, request):
    if request == 'auto':return 'fp16' if device.type == 'cuda' else 'fp32'
    if request in ('fp16','bf16') and device.type != 'cuda':
        raise ValueError('FP16/BF16 mixed precision requires CUDA; choose fp32 for CPU/MPS.')
    if request == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This CUDA device does not support BF16.')
    return request


def autocast(device, precision):
    if device.type != 'cuda' or precision == 'fp32':return nullcontext()
    return torch.autocast('cuda', dtype=torch.float16 if precision=='fp16' else torch.bfloat16)


def scaler_for(precision):
    try:return torch.amp.GradScaler('cuda',enabled=precision=='fp16')
    except (TypeError,AttributeError):return torch.cuda.amp.GradScaler(enabled=precision=='fp16')


def source_files(directory):
    p=Path(directory)
    if not p.exists():raise FileNotFoundError(f'Corpus not found: {p}')
    files=sorted(x for x in (p.rglob('*') if p.is_dir() else [p]) if x.is_file() and x.suffix.lower() in ('.txt','.jsonl'))
    if not files:raise FileNotFoundError(f'No .txt or .jsonl documents in {p}')
    return files


def source_signature(files):
    """Cheap resume guard for files; does not replace checksum-based auditing."""
    s=hashlib.sha256()
    for f in files:
        stat=f.stat()
        s.update(f'{f.resolve()}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    return s.hexdigest()


def iter_documents(files, split='train', limit=0, dedup_window=0):
    """Round-robin domains; 90/5/5 row split, bounded-memory duplicate filter."""
    wanted = {'train':set(range(18)), 'validation':{18}, 'test':{19}}[split]
    def lines(path):
        with path.open('r',encoding='utf-8') as source:
            for index,line in enumerate(source):
                if index % 20 not in wanted:continue
                line=line.strip()
                if not line:continue
                if path.suffix.lower() == '.jsonl':
                    item=json.loads(line)
                    if not isinstance(item,dict):continue
                    if isinstance(item.get('text'),str):text=item['text']
                    else:text='\n'.join(str(item.get(k,'')) for k in ('instruction','response'))
                else:text=line
                if text.strip():yield text
    streams=deque(map(iter, (lines(p) for p in files)))
    history=deque();seen=set();emitted=0
    while streams:
        iterator=streams.popleft()
        try:doc=next(iterator)
        except StopIteration:continue
        streams.append(iterator)
        if dedup_window:
            fingerprint=hashlib.blake2b(doc.encode('utf-8'),digest_size=16).digest()
            if fingerprint in seen:continue
            seen.add(fingerprint);history.append(fingerprint)
            if len(history)>dedup_window:seen.remove(history.popleft())
        yield doc
        emitted+=1
        if limit and emitted>=limit:return


def token_batches(documents, tokenizer, sequence_length, batch_size, skip=0, max_record_tokens=65536):
    """Dense, document-separated next-token blocks with deterministic replay for resume.

    Skip means complete microbatches already consumed. A single EOS separates
    documents; every non-padding label is an actual next-token target.
    """
    if sequence_length<8 or batch_size<1:raise ValueError('Invalid sequence length/batch size')
    tokens=[];pending=[];index=0
    eos=tokenizer.eos_id()
    pad=tokenizer.pad_id()
    if eos<0 or pad<0:raise ValueError('Tokenizer must define EOS and PAD ids.')
    for document in documents:
        ids=tokenizer.encode(document,out_type=int)
        if not ids:continue
        tokens.extend(ids[:max_record_tokens]);tokens.append(eos)
        while len(tokens)>sequence_length:
            x=tokens[:sequence_length]
            y=tokens[1:sequence_length+1]
            del tokens[:sequence_length]
            pending.append((x,y))
            if len(pending)==batch_size:
                if index>=skip:yield tensor_batch(pending,sequence_length,pad)
                index+=1;pending=[]
    if len(tokens)>1:
        pending.append((tokens[:-1],tokens[1:]))
    if pending and index>=skip:yield tensor_batch(pending,sequence_length,pad)


def tensor_batch(pairs,length,pad):
    x=torch.full((len(pairs),length),pad,dtype=torch.long)
    y=torch.full_like(x,-100)
    for i,(a,b) in enumerate(pairs):
        n=min(len(a),length)
        x[i,:n]=torch.tensor(a[:n])
        y[i,:n]=torch.tensor(b[:n])
    return x,y


def evaluate_validation(model, paths, tokenizer, device, precision, sequence_length,
                        batch_size=1, max_batches=10, max_record_tokens=65536):
    """Token-weighted loss on held-out rows; not an independent benchmark."""
    if batch_size < 1 or max_batches < 1:
        raise ValueError('Validation settings must be positive')
    model.eval()
    total_loss, token_count = 0.0, 0
    try:
        documents = iter_documents(paths, split='validation')
        batches = token_batches(documents, tokenizer, sequence_length,
                                batch_size, max_record_tokens=max_record_tokens)
        with torch.inference_mode():
            for index, (x, y) in enumerate(batches):
                if index >= max_batches:
                    break
                active = int((y != -100).sum().item())
                if not active:
                    continue
                x, y = x.to(device), y.to(device)
                with autocast(device, precision):
                    _, loss = model(x, y)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Non-finite validation loss')
                total_loss += float(loss.item()) * active
                token_count += active
    finally:
        model.train()
    return (total_loss / token_count if token_count else float('nan')), token_count


def validate_model_size():
    with torch.device('meta'):
        model=MiniAI(ModelConfig())
    count=sum(x.numel() for x in model.parameters())
    assert 99_900_000 <= count <= 100_200_000,(count,'unexpected model size')
    return count


def get_config(args):
    return ModelConfig(vocab_size=args.vocab_size,context_length=128,n_layer=2,n_head=4,d_model=64,d_ff=128,dropout=0.0) if args.tiny else ModelConfig()


def train(args):
    if min(args.epochs,args.batch_size,args.grad_accum,args.save_every,args.log_every,args.eval_batches,args.eval_batch_size)<1:raise ValueError('epochs/batch/accum/save/log must be positive')
    if args.sequence_length<8 or args.lr<=0 or args.warmup_steps<0 or args.lr_decay_steps<=args.warmup_steps:
        raise ValueError('Invalid context length or optimizer schedule')
    if not 0 <= args.min_lr_ratio <= 1:raise ValueError('min-lr-ratio must be in 0..1')
    paths=source_files(args.corpus)
    sig=source_signature(paths)
    import sentencepiece as spm
    tokenpath=Path(args.tokenizer)
    if not tokenpath.is_file():raise FileNotFoundError(f'Tokenizer not found: {tokenpath}; run miniai.py prepare')
    tok=spm.SentencePieceProcessor(model_file=str(tokenpath))
    torch.manual_seed(args.seed)
    cfg=get_config(args)
    if tok.vocab_size()>cfg.vocab_size:
        raise ValueError(f'Tokenizer vocab {tok.vocab_size()} exceeds model vocab {cfg.vocab_size}')
    if args.sequence_length>cfg.context_length:raise ValueError('Sequence length exceeds model context')
    device=device_for(args.device)
    precision=precision_for(device,args.precision)
    if device.type=='cpu':torch.set_num_threads(min(8,os.cpu_count() or 1))
    model=MiniAI(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scaler=scaler_for(precision)
    output=Path(args.output);output.mkdir(exist_ok=True,parents=True)
    tokenhash=hashlib.sha256(tokenpath.read_bytes()).hexdigest()
    step=0;start_epoch=0;skip=0;tokens_seen=0;best_loss=float('inf')
    settings={'sequence_length':args.sequence_length,'batch_size':args.batch_size,'grad_accum':args.grad_accum,'lr':args.lr,'weight_decay':args.weight_decay,'warmup_steps':args.warmup_steps,'lr_decay_steps':args.lr_decay_steps,'min_lr_ratio':args.min_lr_ratio,'precision':precision,'dedup_window':args.dedup_window,'max_record_tokens':args.max_record_tokens,'max_records':args.max_records}
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=True)
        if ck['config']!=asdict(cfg) or ck.get('training_mode')!='causal_pretraining':raise ValueError('Incompatible model or checkpoint stage')
        if ck.get('corpus_signature')!=sig or ck.get('tokenizer_sha256')!=tokenhash or ck.get('settings')!=settings:
            raise ValueError('Corpus, tokenizer or training options changed since checkpoint.')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        for state in optimizer.state.values():
            for key,v in state.items():
                if isinstance(v,torch.Tensor):state[key]=v.to(device)
        if ck.get('scaler'):scaler.load_state_dict(ck['scaler'])
        step=ck['step'];start_epoch=ck['epoch'];skip=ck['micro_cursor'];tokens_seen=ck.get('tokens_seen',0);best_loss=ck.get('best_loss',best_loss)
        if 'rng_state' in ck:torch.set_rng_state(ck['rng_state'])
        print(f'BY MATEN/T | resume epoch={start_epoch} batch={skip} optimizer_step={step}',flush=True)

    def save(epoch,cursor):
        checkpoint={'created_by':'MATEN/T','format_version':'1.1','training_mode':'causal_pretraining',
                    'config':asdict(cfg),'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'scaler':scaler.state_dict(),'step':step,'epoch':epoch,'micro_cursor':cursor,
                    'tokens_seen':tokens_seen,'best_loss':best_loss,'rng_state':torch.get_rng_state(),
                    'corpus_signature':sig,'tokenizer_sha256':tokenhash,'settings':settings}
        temp=output/'last.tmp'
        torch.save(checkpoint,temp)
        os.replace(temp,output/'last.pt')
        print(f'BY MATEN/T | checkpoint {output/"last.pt"}; step={step}, tokens={tokens_seen:,}',flush=True)

    def lr_for(i):
        if args.warmup_steps and i<args.warmup_steps:return args.lr*(i+1)/args.warmup_steps
        progress=min(1.0,max(0.0,(i-args.warmup_steps)/(args.lr_decay_steps-args.warmup_steps)))
        return args.lr*(args.min_lr_ratio+(1-args.min_lr_ratio)*(0.5+0.5*math.cos(math.pi*progress)))

    def update(denominator):
        nonlocal step
        # Scale the short final accumulation group to preserve the mean gradient.
        if denominator != args.grad_accum:
            for p in model.parameters():
                if p.grad is not None:p.grad.mul_(args.grad_accum/denominator)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
        rate=lr_for(step)
        for group in optimizer.param_groups:group['lr']=rate
        before=scaler.get_scale()
        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        if precision=='fp16' and scaler.get_scale()<before:return False
        step+=1
        return True

    print(f'BY MATEN/T | v1.1 model={sum(p.numel() for p in model.parameters()):,} context={cfg.context_length} precision={precision} docs={len(paths)}',flush=True)
    for epoch in range(start_epoch,args.epochs):
        model.train();optimizer.zero_grad(set_to_none=True)
        acc=0;cursor=skip if epoch==start_epoch else 0;last_loss=float('nan')
        documents=iter_documents(paths,split='train',limit=args.max_records,dedup_window=args.dedup_window)
        batches=token_batches(documents,tok,args.sequence_length,args.batch_size,skip=cursor,max_record_tokens=args.max_record_tokens)
        for x,y in batches:
            if not bool((y!=-100).any()):continue
            active=int((y!=-100).sum().item())
            x=x.to(device);y=y.to(device)
            with autocast(device,precision):_,loss=model(x,y)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss; try lower LR or FP32')
            scaler.scale(loss/args.grad_accum).backward()
            acc+=1;cursor+=1;tokens_seen+=active;last_loss=float(loss.item())
            if acc<args.grad_accum:continue
            acc=0
            if update(args.grad_accum):
                if step%args.log_every==0:print(f'epoch={epoch+1} step={step} loss={last_loss:.5f} tokens_seen={tokens_seen:,}',flush=True)
                if step%args.save_every==0:save(epoch,cursor)
                if args.max_steps and step>=args.max_steps:
                    save(epoch,cursor)
                    return output/'last.pt'
        if acc:update(acc)
        validation_loss, validation_tokens = evaluate_validation(
            model, paths, tok, device, precision, args.sequence_length,
            batch_size=args.eval_batch_size, max_batches=args.eval_batches,
            max_record_tokens=args.max_record_tokens)
        if math.isfinite(validation_loss):
            best_loss = min(best_loss, validation_loss)
        print(f'BY MATEN/T | epoch={epoch+1} validation_loss={validation_loss:.6f} '
              f'validation_tokens={validation_tokens:,} best_validation_loss={best_loss:.6f}',flush=True)
        save(epoch+1,0)
        skip=0
    print(f'BY MATEN/T | pretraining epochs finished, steps={step}, tokens_seen={tokens_seen:,}',flush=True)
    if not args.no_export:
        from export_model import export_checkpoint
        export_checkpoint(output/'last.pt',output/'release',tokenpath,precision='fp16')
    return output/'last.pt'


def parser():
    p=argparse.ArgumentParser(description='MATEN/T MiniAI v1.1: true causal language-model pretraining')
    p.add_argument('--corpus',default=str(ROOT/'datasets'))
    p.add_argument('--tokenizer',default=str(ROOT/'tokenizer'/'tokenizer.model'))
    p.add_argument('--output',default=str(ROOT/'pretrain_checkpoints'))
    p.add_argument('--device',choices=['auto','cuda','mps','cpu'],default='auto')
    p.add_argument('--precision',choices=['auto','fp16','bf16','fp32'],default='auto')
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=8)
    p.add_argument('--sequence-length',type=int,default=512)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--weight-decay',type=float,default=0.1)
    p.add_argument('--warmup-steps',type=int,default=100)
    p.add_argument('--lr-decay-steps',type=int,default=50000)
    p.add_argument('--min-lr-ratio',type=float,default=0.1)
    p.add_argument('--grad-clip',type=float,default=1.0)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--log-every',type=int,default=20)
    p.add_argument('--eval-batches',type=int,default=10,help='Held-out validation batches per epoch')
    p.add_argument('--eval-batch-size',type=int,default=1)
    p.add_argument('--max-steps',type=int,default=0,help='0 means full epochs; positive values only for debugging')
    p.add_argument('--max-records',type=int,default=0,help='0 means all rows')
    p.add_argument('--max-record-tokens',type=int,default=65536)
    p.add_argument('--dedup-window',type=int,default=50000,help='Bounded exact duplicate filter; not a full corpus dedup')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',default='')
    p.add_argument('--vocab-size',type=int,default=16384,help='Used only with --tiny for pipeline tests')
    p.add_argument('--tiny',action='store_true',help='Tiny test model, not the real 100M checkpoint')
    p.add_argument('--no-export',action='store_true')
    return p

if __name__=='__main__':train(parser().parse_args())
