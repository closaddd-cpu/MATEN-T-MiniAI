# Model Card — MATEN/T MiniAI v1

**BY MATEN/T** · **M/ License v1.0**

## Architecture
- Decoder-only Transformer with 10 blocks, 12 attention heads, hidden width 768 and FFN width 3240.
- **86,045,328 trainable parameters**.
- **6,144-token context** with Rotary Position Embeddings (RoPE).
- SentencePiece BPE, 16,384 vocabulary items.
- FP16 autocast with GradScaler on supported CUDA devices; optional BF16; FP32 otherwise.
- AdamW training, warmup, cosine learning-rate schedule, gradient accumulation, gradient clipping, streamed packed examples, token accounting, save/resume.

## Compatibility and availability
This architecture is not checkpoint-compatible with the earlier 2,048-token learned-absolute-position model. This repository does **not** provide trained model weights or a prepared tokenizer. Those must be generated and trained.

## Dataset and limitations
Up to 9.4M procedurally generated JSONL examples across 94 domains. The examples are short, heavily template-based and not independently human reviewed. A 6K context capacity does not mean 6K reading accuracy. Real fluency in supported languages, expert code generation, reliable scientific reasoning and robust cybersecurity judgments are **not demonstrated**.

## Evaluation
Source-level CPU smoke tests are not a substitute for full-run convergence and independent benchmarks. Compare validation losses across checkpoints and use held-out, naturally authored test sets. Do not publish an untrained model as an effective assistant.
