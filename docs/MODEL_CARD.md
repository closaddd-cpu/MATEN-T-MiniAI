# Model Card — MATEN/T MiniAI v1.1

**BY MATEN/T — M/ License v1.0**

## Architecture
- Decoder-only causal Transformer, 100,062,768 base trainable parameters.
- 10 Transformer layers, hidden width 768, 12 attention heads, FFN width 4152.
- 6,144-token maximum context with RoPE and 16,384 SentencePiece BPE vocabulary pieces.
- Next-token causal pretraining over raw text and JSONL, with streaming packing and EOS document boundaries.
- Separate LoRA SFT adapters for attention query/key/value and output projections.
- PyTorch training with AdamW, gradient clipping, warmup/decay, FP16 AMP and atomic checkpoints.
- SafeTensors FP16 export with config and true SentencePiece files; LoRA can be merged for inference.

## Release status
The public repository contains runnable training code and a dataset generator, **not pretrained 100M weights**. The earlier 86M checkpoints cannot be loaded into this changed architecture. The 6K context size is an architectural limit, not a measured retrieval/attention benchmark.

## Training limitations
The 9.4M generated records are mostly short synthetic templates; they are insufficient evidence of fluent conversation, expert programming or deep scientific reasoning. Next-token learning can result in memorization. Using genuine licensed training text, removing duplicates and evaluating on human-reviewed, held-out data is required to assess understanding and generalization.

## Safety and provenance
Do not use generated material for high-stakes decisions without review. LoRA adapters are base-model-specific and must match model architecture and tokenizer. The project uses a custom M/ License (not OSI approved).
