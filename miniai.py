# BY MATEN/T | MATEN/T MiniAI v1 | M/ License v1.0
"""Train a causal 86M decoder model from random weights; chat and local UI.
Usage: python miniai.py train|chat|serve|evaluate|prepare|info
"""
from __future__ import annotations
from brand_guard import enforce_attribution
enforce_attribution()
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
import random
from pathlib import Path
import torch
from model import ModelConfig, MiniAI
from build_datasets import FILES
from catalog import NATURAL

ROOT=Path(__file__).resolve().parent
TOK=ROOT/'tokenizer'/'tokenizer.model'


def load_tokenizer():
    import sentencepiece as spm
    if not TOK.is_file(): raise FileNotFoundError('Tokenizer missing; run python miniai.py prepare')
    return spm.SentencePieceProcessor(model_file=str(TOK))

def device_for(name):
    if name=='auto':
        if torch.cuda.is_available(): return torch.device('cuda')
        if hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): return torch.device('mps')
        return torch.device('cpu')
    if name=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; check NVIDIA driver and CUDA-compatible PyTorch build.')
    if name=='mps' and not (hasattr(torch.backends,'mps') and torch.backends.mps.is_available()):raise RuntimeError('MPS unavailable')
    return torch.device(name)

def resolve_precision(device, choice='auto'):
    """FP16 mixed precision on CUDA; safely stay in FP32 on CPU/MPS."""
    if choice not in ('auto', 'fp16', 'bf16', 'fp32'):
        raise ValueError('precision must be auto, fp16, bf16 or fp32')
    if choice == 'auto':
        return 'fp16' if device.type == 'cuda' else 'fp32'
    if choice in ('fp16', 'bf16') and device.type != 'cuda':
        raise ValueError(f'{choice} mixed-precision training requires a CUDA device; use --precision auto or fp32.')
    if choice == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 is not supported on this CUDA device; choose --precision fp16.')
    return choice


def cast_for(device, precision='auto'):
    mode=resolve_precision(device, precision)
    if mode == 'fp32': return nullcontext()
    return torch.autocast(device_type='cuda',dtype=torch.float16 if mode=='fp16' else torch.bfloat16)


def make_scaler(device, precision):
    """Dynamic gradient scaling avoids FP16 gradient underflow."""
    enabled=device.type=='cuda' and precision=='fp16'
    try: return torch.amp.GradScaler('cuda',enabled=enabled)
    except (AttributeError,TypeError): return torch.cuda.amp.GradScaler(enabled=enabled)


def learning_rate_at(step, base_lr, warmup, decay, min_ratio=0.1, schedule='cosine'):
    """Warmup and bounded cosine decay; step counts *optimizer* updates."""
    if base_lr <= 0 or warmup < 0 or decay < 1 or not (0 <= min_ratio <= 1):
        raise ValueError('Invalid learning-rate schedule settings')
    if schedule not in ('cosine','constant'):raise ValueError('Invalid LR schedule')
    if warmup and step < warmup:
        return base_lr * (step+1)/warmup
    if schedule=='constant':return base_lr
    if decay <= warmup:raise ValueError('--lr-decay-steps must exceed --warmup-steps for cosine decay')
    progress=min(1.0,max(0.0,(step-warmup)/(decay-warmup)))
    return base_lr*(min_ratio+(1-min_ratio)*0.5*(1+math.cos(math.pi*progress)))

def file_rows(paths, split, max_rows=0):
    """Stream without loading all the JSONL corpus into RAM.
    Split is stable per shard. train=90%, validation=5%, test=5%.
    """
    from collections import deque
    sent=0
    wanted = {'train':set(range(18)), 'validation':{18}, 'test':{19}}[split]
    def per_shard(path):
        with path.open('r',encoding='utf-8') as file:
            for idx,line in enumerate(file):
                if idx%20 in wanted:yield json.loads(line)
    # Round-robin each domain: short/partial runs also see ALL selected topics.
    queue=deque([iter(per_shard(path)) for path in paths])
    while queue:
        source=queue.popleft()
        try:row=next(source)
        except StopIteration:continue
        yield row;sent+=1
        if max_rows and sent>=max_rows:return
        queue.append(source)

def data_files(folder, domains=None):
    domains = domains or FILES
    paths=[Path(folder)/(d+'.jsonl') for d in domains]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError('Missing dataset files: '+', '.join(missing[:5])+'; run python build_datasets.py')
    return paths

def example(row, tok, limit):
    user=tok.piece_to_id('<|user|>')
    assistant=tok.piece_to_id('<|assistant|>')
    end=tok.piece_to_id('<|end|>')
    if any(x==tok.unk_id() for x in [user,assistant,end]):raise ValueError('Missing chat marker in tokenizer; use python miniai.py prepare')
    question=tok.encode(row['instruction'],out_type=int)
    answer=tok.encode(row['response'],out_type=int)+[end]
    if len(answer)>limit-4:answer=answer[:limit-4]+[end]
    room=limit-3-len(answer)
    if len(question)>room:question=question[-room:] if room>0 else []
    ids=[tok.bos_id(),user]+question+[assistant]+answer
    labels=[-100]*(len(question)+3)+answer
    return ids[:-1],labels[1:]

def batches(rows, tokenizer, seq_len, batch_size, pad_id, skip=0):
    cache=[]; batch_no=0
    for row in rows:
        cache.append(example(row,tokenizer,seq_len))
        if len(cache)==batch_size:
            if batch_no>=skip:yield collate(cache,pad_id)
            batch_no+=1;cache=[]
    if cache and batch_no>=skip: yield collate(cache,pad_id)

def packed_batches(rows, tokenizer, seq_len, batch_size, pad_id, skip=0):
    """Pack variable-length examples into dense blocks with streaming memory use."""
    block_x,block_y,pending=[],[],[]
    batch_no=0
    for row in rows:
        ids,targets=example(row,tokenizer,seq_len)
        cursor=0
        while cursor < len(ids):
            take=min(seq_len-len(block_x),len(ids)-cursor)
            block_x.extend(ids[cursor:cursor+take])
            block_y.extend(targets[cursor:cursor+take])
            cursor+=take
            if len(block_x)==seq_len:
                pending.append((block_x,block_y))
                block_x,block_y=[],[]
                if len(pending)==batch_size:
                    if batch_no>=skip: yield collate(pending,pad_id)
                    batch_no+=1;pending=[]
    if block_x:pending.append((block_x,block_y))
    if pending and batch_no>=skip:yield collate(pending,pad_id)


def collate(pairs,pad_id):
    length=max(len(x) for x,_ in pairs)
    x=torch.full((len(pairs),length),pad_id,dtype=torch.long)
    y=torch.full_like(x,-100)
    for i,(ids,labels) in enumerate(pairs):
        x[i,:len(ids)]=torch.tensor(ids)
        y[i,:len(labels)]=torch.tensor(labels)
    return x,y

def miniature():
    return ModelConfig(n_layer=2,n_head=4,d_model=128,d_ff=384,context_length=128)

def read_checkpoint(path):
    # torch.load may deserialize objects: only load files you trust.
    return torch.load(path,map_location='cpu',weights_only=False)

def evaluate(model,paths,tok,device,limit,bsize,max_batches,split='validation',precision='auto'):
    model.eval(); losses=[]
    with torch.inference_mode():
        rows=file_rows(paths,split,max_rows=max_batches*bsize)
        for x,y in batches(rows,tok,limit,bsize,tok.pad_id()):
            x=x.to(device);y=y.to(device)
            with cast_for(device,precision):_,loss=model(x,y)
            losses.append(float(loss.item()))
    model.train()
    return sum(losses)/len(losses) if losses else float('nan')

def train(args):
    tok=load_tokenizer()
    paths=data_files(args.data,args.domains)
    device=device_for(args.device)
    cfg=miniature() if args.tiny else ModelConfig()
    if args.sequence_length>cfg.context_length:raise ValueError('Requested sequence length exceeds architecture context.')
    seq_len=min(args.sequence_length,cfg.context_length)
    if seq_len<16:raise ValueError('sequence length must be at least 16')
    if args.batch_size<1 or args.grad_accum<1 or args.epochs<1:raise ValueError('batch-size, grad-accum, epochs must be positive')
    precision=resolve_precision(device,args.precision)
    schedule=(args.lr_schedule, args.warmup_steps, args.lr_decay_steps, args.min_lr_ratio, args.lr)
    learning_rate_at(0,args.lr,args.warmup_steps,args.lr_decay_steps,args.min_lr_ratio,args.lr_schedule)  # validate early
    if args.lr_schedule=='cosine' and args.lr_decay_steps<=args.warmup_steps:raise ValueError('LR decay steps must exceed warmup steps')
    if args.log_every<1 or args.save_every<1:raise ValueError('log-every and save-every must be positive')
    print(f'BY MATEN/T | device={device}, precision={precision} (mixed precision; FP32 master weights), context_capacity={cfg.context_length}, training_sequence={seq_len}',flush=True)
    print('IMPORTANT | this is training from random initialization on synthetic data; output quality is not guaranteed.',flush=True)
    torch.manual_seed(args.seed);random.seed(args.seed)
    if device.type=='cpu':torch.set_num_threads(min(os.cpu_count() or 1,8))
    model=MiniAI(cfg).to(device)
    print('Trainable parameters:',format(sum(p.numel() for p in model.parameters()),','),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.1)
    scaler=make_scaler(device,precision)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    start_epoch=0;micro_cursor=0;step=0;tokens_seen=0;supervised_tokens_seen=0
    if args.resume:
        ck=read_checkpoint(args.resume)
        if ck['config']!=asdict(cfg):raise ValueError('Architecture mismatch with saved checkpoint')
        if ck.get('domains') != (args.domains or FILES):raise ValueError('Dataset domain choices changed since checkpoint')
        if ck.get('batch_size')!=args.batch_size or ck.get('grad_accum')!=args.grad_accum or ck.get('sequence_length')!=seq_len:
            raise ValueError('Batch / accumulation / sequence-length settings changed since checkpoint')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);
        if ck.get('lr_schedule_config') and tuple(ck['lr_schedule_config'])!=schedule:
            raise ValueError('Learning-rate schedule settings changed since checkpoint')
        if ck.get('scaler') and precision=='fp16':scaler.load_state_dict(ck['scaler'])
        start_epoch=ck['epoch'];micro_cursor=ck['micro_cursor'];step=ck['step']
        if ck.get('pack') != args.pack: raise ValueError('Packing setting changed since checkpoint')
        tokens_seen=ck.get('tokens_seen',0);supervised_tokens_seen=ck.get('supervised_tokens_seen',0)
        print(f'Resuming at epoch {start_epoch+1}, microbatch index {micro_cursor}, optimizer step {step}',flush=True)
    # Model/optimizer remain in device memory; input corpus is streamed through Python.
    def save(epoch, cursor):
        obj={'created_by':'MATEN/T','license':'M/ License v1.0','config':asdict(cfg),
             'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
             'epoch':epoch,'micro_cursor':cursor,'step':step,'domains':args.domains or FILES,
             'batch_size':args.batch_size,'grad_accum':args.grad_accum,'sequence_length':seq_len,
             'precision':precision,'lr_schedule_config':schedule,
             'pack':args.pack,'tokens_seen':tokens_seen,'supervised_tokens_seen':supervised_tokens_seen}
        tmp=out/'last.tmp';torch.save(obj,tmp);os.replace(tmp,out/'last.pt')
        print(f'Checkpoint saved at step {step} -> {out/"last.pt"}',flush=True)
    for epoch in range(start_epoch,args.epochs):
        opt.zero_grad(set_to_none=True); accumulated=0;loss_total=0.0;micro=0
        path_order=paths[epoch%len(paths):]+paths[:epoch%len(paths)]
        rows=file_rows(path_order,'train',max_rows=args.limit_samples)
        skip=micro_cursor if epoch==start_epoch else 0
        batch_source=packed_batches if args.pack else batches
        for x,y in batch_source(rows,tok,seq_len,args.batch_size,tok.pad_id(),skip=skip):
            micro+=1
            tokens_seen+=int((x!=tok.pad_id()).sum().item())
            supervised_tokens_seen+=int((y!=-100).sum().item())
            x=x.to(device);y=y.to(device)
            with cast_for(device,precision):_,loss=model(x,y)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            scaler.scale(loss/args.grad_accum).backward();accumulated+=1;loss_total+=float(loss.item())
            if accumulated<args.grad_accum:continue
            scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            lr=learning_rate_at(step,args.lr,args.warmup_steps,args.lr_decay_steps,args.min_lr_ratio,args.lr_schedule)
            for group in opt.param_groups:group['lr']=lr
            before_scale=scaler.get_scale()
            scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            # On FP16 overflow GradScaler skips the optimizer update. Do not count it as a trained step.
            skipped=precision=='fp16' and scaler.get_scale()<before_scale
            accumulated=0
            if skipped:
                print('BY MATEN/T | FP16 overflow: optimizer step skipped; scaler reduced',flush=True)
                continue
            step+=1
            if step%args.log_every==0: print(f'epoch={epoch+1} step={step} lr={lr:.3g} precision={precision} mean_loss={loss_total/max(micro,1):.5f} tokens_seen={tokens_seen:,} supervised_targets={supervised_tokens_seen:,}',flush=True)
            if step%args.save_every==0:save(epoch,skip+micro)
            if args.max_steps and step>=args.max_steps:save(epoch,skip+micro);return
        if accumulated:
            # Correct partial-group gradient scale before the final update.
            factor=args.grad_accum/accumulated
            for p in model.parameters():
                if p.grad is not None:p.grad.mul_(factor)
            scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            lr=learning_rate_at(step,args.lr,args.warmup_steps,args.lr_decay_steps,args.min_lr_ratio,args.lr_schedule)
            for group in opt.param_groups:group['lr']=lr
            before_scale=scaler.get_scale()
            scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            if not (precision=='fp16' and scaler.get_scale()<before_scale):step+=1
        val=evaluate(model,paths,tok,device,seq_len,args.eval_batch_size,args.eval_batches,precision=precision)
        print(f'Finished epoch={epoch+1}, validation loss={val:.5f}, tokens_seen={tokens_seen:,}, supervised_targets={supervised_tokens_seen:,}',flush=True)
        save(epoch+1,0)
        micro_cursor=0
    print('Training epochs complete. This does NOT imply reliable factual performance.',flush=True)

def loaded_model(checkpoint,device):
    ck=read_checkpoint(checkpoint);cfg=ModelConfig(**ck['config'])
    model=MiniAI(cfg).to(device);model.load_state_dict(ck['model']);model.eval()
    return model,cfg

def infer(model,cfg,tok,device,text,max_new_tokens,temperature):
    end=tok.piece_to_id('<|end|>')
    ids=[tok.bos_id(),tok.piece_to_id('<|user|>')]+tok.encode(text,out_type=int)+[tok.piece_to_id('<|assistant|>')]
    ids=ids[-max(1,cfg.context_length-max_new_tokens):]
    tensor=torch.tensor([ids],device=device)
    with torch.inference_mode():out=model.generate(tensor,max_new_tokens=max_new_tokens,temperature=temperature,top_k=30,stop_id=end)
    answer=out[0,len(ids):].tolist()
    if end in answer:answer=answer[:answer.index(end)]
    return tok.decode(answer).strip()

def chat(args):
    device=device_for(args.device);model,cfg=loaded_model(args.checkpoint,device);tok=load_tokenizer()
    print('BY MATEN/T | /exit to quit. Experimental weights; verify important claims.')
    while True:
        try:s=input('You> ').strip()
        except (KeyboardInterrupt,EOFError):break
        if s.lower() in ('/exit','/quit'):break
        if s:print('MiniAI>',infer(model,cfg,tok,device,s,args.max_new_tokens,args.temperature))

def serve(args):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    device=device_for(args.device);model,cfg=loaded_model(args.checkpoint,device);tok=load_tokenizer()
    from threading import Lock
    lock=Lock()
    index=(ROOT/'ui'/'index.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!='/':self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(index)));self.end_headers();self.wfile.write(index)
        def do_POST(self):
            if self.path!='/api/chat':self.send_error(404);return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if size<=0 or size>16000:raise ValueError('Invalid request length')
                body=json.loads(self.rfile.read(size).decode('utf-8'))
                question=body.get('message','')
                if not isinstance(question,str) or not question.strip() or len(question)>4000:raise ValueError('Message must contain 1..4000 characters')
                language=body.get('language','auto')
                if language != 'auto' and language not in NATURAL:raise ValueError('Unsupported language selection')
                if language != 'auto':
                    preferred = NATURAL[language][0]
                    question = f'Reply in {preferred}.\n' + question
                with lock:reply=infer(model,cfg,tok,device,question, min(256,cfg.context_length//4),0.7)
                payload=json.dumps({'answer':reply or '[No output generated]'},ensure_ascii=False).encode('utf-8')
                self.send_response(200)
            except (ValueError,TypeError,KeyError) as ex:
                payload=json.dumps({'error':str(ex)}).encode();self.send_response(400)
            except Exception:
                payload=json.dumps({'error':'Generation failed; see server logs.'}).encode();self.send_response(500)
            self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    if args.host not in ('127.0.0.1','localhost','::1'):raise ValueError('Local-only chat UI: host must be loopback')
    print(f'BY MATEN/T | Chat UI on http://127.0.0.1:{args.port} (local only)',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

def prepare(args):
    import sentencepiece as spm
    paths=data_files(args.data,args.domains)
    TOK.parent.mkdir(exist_ok=True,parents=True)
    corpus=TOK.parent/'corpus_for_tokenizer.txt'
    count=0
    with corpus.open('w',encoding='utf-8') as output:
        for path in paths:
            for row in file_rows([path],'train',max_rows=args.samples_per_domain):
                output.write(row['instruction'].replace('\n',' ')+'\n')
                output.write(row['response'].replace('\n',' ')+'\n');count+=1
    spm.SentencePieceTrainer.train(input=str(corpus),model_prefix=str(TOK.with_suffix('')),
                                  vocab_size=16384,model_type='bpe',character_coverage=1.,hard_vocab_limit=False,
                                  bos_id=1,eos_id=2,pad_id=3,unk_id=0,
                                  user_defined_symbols=['<|user|>','<|assistant|>','<|end|>'],
                                  num_threads=min(os.cpu_count() or 2,8),max_sentence_length=8192)
    corpus.unlink(missing_ok=True)
    print('BY MATEN/T | trained tokenizer from',count,'train-only samples. Model:',TOK)

def parser():
    p=argparse.ArgumentParser(description='BY MATEN/T | MiniAI v1 86M / 6K context')
    sub=p.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train');tr.add_argument('--data',default=str(ROOT/'datasets'));tr.add_argument('--output',default=str(ROOT/'checkpoints'))
    tr.add_argument('--domains',nargs='+',choices=FILES);tr.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps'])
    tr.add_argument('--epochs',type=int,default=1);tr.add_argument('--batch-size',type=int,default=1);tr.add_argument('--grad-accum',type=int,default=8)
    tr.add_argument('--sequence-length',type=int,default=1024);tr.add_argument('--max-steps',type=int,default=0)
    tr.add_argument('--no-pack',action='store_false',dest='pack',help='Disable dense example packing')
    tr.set_defaults(pack=True)
    tr.add_argument('--lr',type=float,default=3e-4);tr.add_argument('--seed',type=int,default=42)
    tr.add_argument('--precision',choices=['auto','fp16','bf16','fp32'],default='auto',
                    help='auto uses FP16 autocast+GradScaler on CUDA, FP32 on CPU/MPS')
    tr.add_argument('--lr-schedule',choices=['cosine','constant'],default='cosine')
    tr.add_argument('--warmup-steps',type=int,default=100)
    tr.add_argument('--lr-decay-steps',type=int,default=10000,
                    help='Set to expected total optimizer steps; decay to --min-lr-ratio')
    tr.add_argument('--min-lr-ratio',type=float,default=0.1)
    tr.add_argument('--save-every',type=int,default=500);tr.add_argument('--log-every',type=int,default=20)
    tr.add_argument('--eval-batches',type=int,default=10);tr.add_argument('--eval-batch-size',type=int,default=2)
    tr.add_argument('--resume',default='');tr.add_argument('--limit-samples',type=int,default=0)
    tr.add_argument('--tiny',action='store_true',help='Smaller model ONLY for pipeline tests')
    tok=sub.add_parser('prepare');tok.add_argument('--data',default=str(ROOT/'datasets'));tok.add_argument('--domains',nargs='+',choices=FILES);tok.add_argument('--samples-per-domain',type=int,default=1000)
    for command in ('chat','serve'):
        cp=sub.add_parser(command);cp.add_argument('--checkpoint',default=str(ROOT/'checkpoints'/'last.pt'))
        cp.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
        if command=='chat':cp.add_argument('--max-new-tokens',type=int,default=128);cp.add_argument('--temperature',type=float,default=0.7)
        else:cp.add_argument('--port',type=int,default=8765);cp.add_argument('--host',default='127.0.0.1')
    ev=sub.add_parser('evaluate');ev.add_argument('--checkpoint',default=str(ROOT/'checkpoints'/'last.pt'))
    ev.add_argument('--data',default=str(ROOT/'datasets'));ev.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps'])
    ev.add_argument('--domains',nargs='+',choices=FILES);ev.add_argument('--split',default='validation',choices=['validation','test'])
    ev.add_argument('--sequence-length',type=int,default=256);ev.add_argument('--batch-size',type=int,default=2)
    ev.add_argument('--max-batches',type=int,default=20)
    ev.add_argument('--precision',choices=['auto','fp16','bf16','fp32'],default='auto')
    sub.add_parser('info')
    return p

if __name__=='__main__':
    args=parser().parse_args()
    if args.command=='train':train(args)
    elif args.command=='chat':chat(args)
    elif args.command=='serve':serve(args)
    elif args.command=='prepare':prepare(args)
    elif args.command=='evaluate':
        device=device_for(args.device); model,cfg=loaded_model(args.checkpoint,device)
        if args.sequence_length>cfg.context_length:raise ValueError('Evaluation sequence exceeds context length')
        loss=evaluate(model,data_files(args.data,args.domains),load_tokenizer(),device,args.sequence_length,args.batch_size,args.max_batches,args.split,precision=args.precision)
        print(f'BY MATEN/T | {args.split} loss: {loss:.6f}; synthetic template overlap may inflate apparent quality.')
    else:
        cfg=ModelConfig()
        print('BY MATEN/T | model_config:',json.dumps(asdict(cfg)), '| CUDA:',torch.cuda.is_available(), '| dataset files:',len(FILES))
