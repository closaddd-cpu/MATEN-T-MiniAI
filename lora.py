# BY MATEN/T | MATEN/T MiniAI v1.1 | M/ License v1.0
"""Small, auditable LoRA adapters for the model's attention projections."""
from __future__ import annotations
from brand_guard import enforce_attribution
enforce_attribution()
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        if rank < 1 or alpha <= 0: raise ValueError('LoRA rank and alpha must be positive')
        self.base=linear
        for p in self.base.parameters():p.requires_grad_(False)
        self.lora_A=nn.Parameter(torch.empty(rank,linear.in_features))
        self.lora_B=nn.Parameter(torch.zeros(linear.out_features,rank))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
        self.scale=alpha/rank
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        update=F.linear(F.linear(self.dropout(x),self.lora_A),self.lora_B)*self.scale
        return self.base(x)+update

    @torch.no_grad()
    def merged_linear(self):
        output=self.base
        update=self.lora_B.float() @ self.lora_A.float()
        output.weight.add_((update*self.scale).to(output.weight.dtype))
        return output


def inject_lora(model,rank=8,alpha=16,dropout=0.05,targets=('qkv','proj')):
    """Freeze the full pretrained base and train LoRA in attention only."""
    for param in model.parameters():param.requires_grad_(False)
    count=0
    for block in model.blocks:
        for name in ('qkv','proj'):
            if name not in targets:continue
            layer=getattr(block.attn,name)
            if not isinstance(layer,nn.Linear):raise TypeError(f'Not a Linear target: attn.{name}')
            setattr(block.attn,name,LoRALinear(layer,rank,alpha,dropout))
            count+=1
    if not count:raise ValueError('No LoRA targets selected')
    return model


def adapter_state(model):
    return {k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()
            if k.endswith('.lora_A') or k.endswith('.lora_B')}


def load_adapter(model,adapter_weights,rank=8,alpha=16,dropout=0.0,merge=False):
    inject_lora(model,rank,alpha,dropout)
    own=model.state_dict()
    expected=set(adapter_state(model))
    if set(adapter_weights)!=expected:raise ValueError('Adapter parameter keys do not match the architecture')
    for key,v in adapter_weights.items():
        if v.shape!=own[key].shape:raise ValueError(f'Adapter shape mismatch: {key}')
        own[key].copy_(v)
    if merge:
        for block in model.blocks:
            for name in ('qkv','proj'):
                layer=getattr(block.attn,name)
                setattr(block.attn,name,layer.merged_linear())
    return model
