# BY MATEN/T — Original causal Transformer trained from scratch.
# M/ License v1.0; PyTorch is a separate dependency under its own license.
from brand_guard import enforce_attribution
# The project checks its original credit before exposing the model, too.
# This is local source-level consistency checking, not tamper-proof DRM.
enforce_attribution()
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    context_length: int = 6144
    n_layer: int = 10
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 3240
    dropout: float = 0.1

    def validate(self):
        if self.d_model % self.n_head: raise ValueError('d_model must be divisible by n_head')
        if self.context_length < 16: raise ValueError('context_length must be >= 16')
        if self.vocab_size < 256: raise ValueError('vocab_size must be >= 256')
        if self.n_layer < 1: raise ValueError('n_layer must be >= 1')


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head=cfg.n_head
        self.d_head=cfg.d_model//cfg.n_head
        if self.d_head % 2: raise ValueError('RoPE requires even head dimensions')
        self.qkv=nn.Linear(cfg.d_model,3*cfg.d_model)
        self.proj=nn.Linear(cfg.d_model,cfg.d_model)
        self.dropout=cfg.dropout
        self.resid_dropout=nn.Dropout(cfg.dropout)

    @staticmethod
    def apply_rope(tensor):
        """Rotary position embeddings for variable-length causal attention."""
        n,d = tensor.size(-2),tensor.size(-1)
        inv = 10000.0 ** (-torch.arange(0,d,2,device=tensor.device,dtype=torch.float32)/d)
        theta = torch.arange(n,device=tensor.device,dtype=torch.float32)[:,None]*inv[None,:]
        cs, sn = theta.cos()[None,None,:,:],theta.sin()[None,None,:,:]
        x0,x1=tensor.float()[...,0::2],tensor.float()[...,1::2]
        out=torch.stack((x0*cs-x1*sn,x0*sn+x1*cs),dim=-1).flatten(-2)
        return out.to(dtype=tensor.dtype)

    def forward(self,x):
        b,t,c=x.shape
        q,k,v=self.qkv(x).split(c,dim=-1)
        q=q.view(b,t,self.n_head,self.d_head).transpose(1,2)
        k=k.view(b,t,self.n_head,self.d_head).transpose(1,2)
        v=v.view(b,t,self.n_head,self.d_head).transpose(1,2)
        q=self.apply_rope(q)
        k=self.apply_rope(k)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout if self.training else 0.,is_causal=True)
        return self.resid_dropout(self.proj(y.transpose(1,2).contiguous().view(b,t,c)))


class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.ln1=nn.LayerNorm(cfg.d_model)
        self.attn=CausalSelfAttention(cfg)
        self.ln2=nn.LayerNorm(cfg.d_model)
        self.fc=nn.Linear(cfg.d_model,cfg.d_ff)
        self.proj=nn.Linear(cfg.d_ff,cfg.d_model)
        self.dropout=nn.Dropout(cfg.dropout)

    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        x=x+self.dropout(self.proj(F.gelu(self.fc(self.ln2(x)))))
        return x


class MiniAI(nn.Module):
    def __init__(self,cfg:ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg=cfg
        self.tok_emb=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.drop=nn.Dropout(cfg.dropout)
        self.blocks=nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f=nn.LayerNorm(cfg.d_model)
        self.head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.head.weight=self.tok_emb.weight  # tied to save parameters
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,mean=0.0,std=0.02)
            if m.bias is not None:nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight,mean=0.0,std=0.02)

    def forward(self,input_ids,targets=None):
        b,t=input_ids.shape
        if t>self.cfg.context_length:raise ValueError(f'Sequence {t} > context {self.cfg.context_length}')
        x=self.drop(self.tok_emb(input_ids))
        for block in self.blocks:x=block(x)
        x=self.ln_f(x)
        logits=self.head(x)
        if targets is None:return logits,None
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),targets.reshape(-1),ignore_index=-100)
        return logits,loss

    @torch.no_grad()
    def generate(self,ids,max_new_tokens=128,temperature=0.8,top_k=30,stop_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            current=ids[:,-self.cfg.context_length:]
            logits,_=self(current)
            logits=logits[:,-1,:]
            if temperature<=0:
                nxt=logits.argmax(dim=-1,keepdim=True)
            else:
                logits=logits/max(temperature,1e-5)
                if top_k>0:
                    vals,_=torch.topk(logits,k=min(top_k,logits.size(-1)))
                    logits=logits.masked_fill(logits<vals[:,-1,None],float('-inf'))
                probs=F.softmax(logits,dim=-1)
                nxt=torch.multinomial(probs,num_samples=1)
            ids=torch.cat((ids,nxt),dim=1)
            if stop_id is not None and nxt.item()==stop_id:break
        return ids


def parameter_count(config=None):
    cfg=config or ModelConfig()
    return sum(p.numel() for p in MiniAI(cfg).parameters())


if __name__=='__main__':
    print(f'BY MATEN/T | parameters: {parameter_count():,}')
