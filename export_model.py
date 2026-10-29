# BY MATEN/T | MATEN/T MiniAI v1.1 | M/ License v1.0
"""Export a trusted checkpoint to a self-describing safetensors release folder.

Separate model weights from optimizer state. Never export an untrained model
as a trained one: the checkpoint must be supplied by a completed training job.
"""
from __future__ import annotations
from brand_guard import enforce_attribution
enforce_attribution()
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import torch
from safetensors.torch import load_file,save_file
from model import ModelConfig,MiniAI

ROOT=Path(__file__).resolve().parent


def safe_state(model,precision='fp16'):
    dtype={'fp16':torch.float16,'fp32':torch.float32,'bf16':torch.bfloat16}[precision]
    weights={}
    for name,param in model.state_dict().items():
        if name=='head.weight':continue  # Tied embedding is stored once.
        tensor=param.detach().to(device='cpu',dtype=dtype).contiguous()
        weights[name]=tensor
    return weights


def restore_weights(model,weights):
    if 'head.weight' in weights:raise ValueError('Expected tied embeddings to be stored only as tok_emb.weight')
    missing,unexpected=model.load_state_dict(weights,strict=False)
    if set(missing)!={'head.weight'} or unexpected:
        raise ValueError(f'Model tensor mismatch: missing={missing} unexpected={unexpected}')
    return model


def export_checkpoint(checkpoint_path,output_dir,tokenizer_path,precision='fp16',adapter_dir=None):
    checkpoint_path=Path(checkpoint_path)
    if not checkpoint_path.exists():raise FileNotFoundError(checkpoint_path)
    ck=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if 'model' not in ck or 'config' not in ck:raise ValueError('Not a model checkpoint')
    if ck.get('training_mode') not in ('causal_pretraining','lora_finetuning','supervised_finetuning'):
        raise ValueError('Unknown checkpoint stage: not exporting unverified weights')
    cfg=ModelConfig(**ck['config']);model=MiniAI(cfg)
    model.load_state_dict(ck['model'])
    if adapter_dir:
        from lora import load_adapter
        folder=Path(adapter_dir)
        config=json.loads((folder/'adapter_config.json').read_text(encoding='utf-8'))
        adapter=load_file(str(folder/'adapter_model.safetensors'),device='cpu')
        model=load_adapter(model,adapter,rank=config['rank'],alpha=config['alpha'],merge=True)
    out=Path(output_dir);out.mkdir(exist_ok=True,parents=True)
    tokenizer_path=Path(tokenizer_path)
    if not tokenizer_path.is_file():raise FileNotFoundError(tokenizer_path)
    import sentencepiece as spm
    tokenizer=spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.vocab_size()>cfg.vocab_size:raise ValueError('Tokenizer size does not match model vocabulary')
    payload=safe_state(model,precision)
    temp=out/'model.tmp.safetensors'
    save_file(payload,str(temp),metadata={'format':'pt','creator':'MATEN/T','version':'1.1','dtype':precision})
    os.replace(temp,out/'model.safetensors')
    config={'model_type':'maten_t_miniai','architectures':['MiniAI'],**asdict(cfg),
            'torch_dtype':{'fp16':'float16','fp32':'float32','bf16':'bfloat16'}[precision],
            'tie_word_embeddings':True,'source_version':'1.1','creator':'MATEN/T',
            'training_mode':ck['training_mode'],'source_step':ck.get('step'),
            'lora_merged':bool(adapter_dir)}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    shutil.copyfile(tokenizer_path,out/'tokenizer.model')
    vocab=tokenizer_path.with_suffix('.vocab')
    if vocab.is_file():shutil.copyfile(vocab,out/'tokenizer.vocab')
    specials={'unk_token':'<unk>','bos_token':'<s>','eos_token':'</s>','pad_token':'<pad>',
              'additional_special_tokens':['<|user|>','<|assistant|>','<|end|>']}
    (out/'special_tokens_map.json').write_text(json.dumps(specials,indent=2)+'\n',encoding='utf-8')
    (out/'tokenizer_config.json').write_text(json.dumps({'tokenizer_class':'SentencePieceProcessor',
        'vocab_size':tokenizer.vocab_size(),'model_max_length':cfg.context_length,
        'bos_token_id':tokenizer.bos_id(),'eos_token_id':tokenizer.eos_id(),
        'pad_token_id':tokenizer.pad_id(),'unk_token_id':tokenizer.unk_id(),
        'additional_special_tokens':specials['additional_special_tokens']},indent=2)+'\n',encoding='utf-8')
    (out/'generation_config.json').write_text(json.dumps({'max_new_tokens':256,'temperature':0.8,
        'top_k':30,'do_sample':True,'eos_token_id':tokenizer.eos_id(),
        'pad_token_id':tokenizer.pad_id()},indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text(f'''# MATEN/T MiniAI v1.1 — Model Export\n\n**BY MATEN/T — M/ License v1.0**\n\nWeights: {sum(p.numel() for p in model.parameters()):,} parameters.\nMaximum context: {cfg.context_length:,} tokens.\n\nThis is a custom PyTorch model, **not a plug-and-play Hugging Face Transformers checkpoint**.\nUse `export_model.py:load_export()` from the original repository to load the\narchitecture and SentencePiece tokenizer. Data quality and training stage must\nbe independently evaluated. `model.safetensors` contains weights only; it\ndoes not contain optimizer state or the training corpus.\n\nRepository: https://github.com/opentaiofficial/MATEN-T-MiniAI\n''',encoding='utf-8')
    digest=hashlib.sha256((out/'model.safetensors').read_bytes()).hexdigest()
    (out/'model.sha256').write_text(f'{digest}  model.safetensors\n',encoding='utf-8')
    print(f'BY MATEN/T | model exported -> {out}; weights_sha256={digest[:16]}...',flush=True)
    return out


def load_export(directory,device='cpu'):
    directory=Path(directory)
    config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
    cfg=ModelConfig(**{k:config[k] for k in asdict(ModelConfig())})
    model=MiniAI(cfg)
    weights=load_file(str(directory/'model.safetensors'),device='cpu')
    restore_weights(model,weights)
    model.to(device).eval()
    return model,cfg


def main():
    p=argparse.ArgumentParser(description='BY MATEN/T | safetensors exporter')
    p.add_argument('--checkpoint',required=True,help='Trusted completed/pretrained PyTorch checkpoint')
    p.add_argument('--output',default='release-v1.1')
    p.add_argument('--tokenizer',default=str(ROOT/'tokenizer'/'tokenizer.model'))
    p.add_argument('--precision',choices=['fp16','fp32','bf16'],default='fp16')
    p.add_argument('--adapter',default=None,help='Optional LoRA adapter directory to merge')
    a=p.parse_args()
    export_checkpoint(a.checkpoint,a.output,a.tokenizer,a.precision,a.adapter)

if __name__=='__main__':main()
