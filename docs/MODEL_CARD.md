# Model Card — MATEN/T MiniAI v1

**BY MATEN/T** · **M/ License v1.0**

- **Architecture:** original decoder-only 10-layer GPT-style Transformer, 86,019,712 parameters, 2,048-token context, 16,384-vocab SentencePiece BPE.
- **Training:** PyTorch from random initialization; cross-entropy, AdamW, gradient accumulation, checkpoint/resume. Python required; C++ data row-counter optional.
- **Model weights:** no pretrained weights in this distribution. Only training machinery, tokenizer and synthetic dataset.
- **Demonstrated:** finite forward/backward and basic save/resume code tests. Full 9.4M-row end-to-end convergence is *not* verified.
- **Not demonstrated:** fluent Kurdish or 20-language dialog, production code generation, deep scientific reasoning, cybersecurity reliability.
- **Safety:** automatically generated text may be erroneous. Human supervision required for consequential science, medical and security uses.
