# MATEN/T MiniAI v1.1 — Verification and quality report

**BY MATEN/T — M/ License v1.0**

## Verified locally for the v1.1 source update
- Architecture parameter count: **100,062,768**, using a meta-device model to avoid allocating full weights.
- Maximum configuration context: **6,144 tokens**, with RoPE positional encoding.
- Source unit tests cover next-token alignment, train/validation separation, LoRA trainable weights, tied embedding export and basic model gradients.
- A **tiny, separate** CPU end-to-end test exercised causal pretraining, checkpoint save/resume, SafeTensors export/load, LoRA fine-tuning and merged export.

## Not verified
- A full 100M pretraining epoch on 8.46M generated rows has **not** been run here.
- No meaningful multilingual competence or generalization benchmark is claimed.
- Full 6K optimization at production scale, all CUDA hardware combinations and massive custom corpora have not been benchmarked.
- Generated data is synthetic and may contain template overlap across dataset splits.

## Quality improvements for real training
1. Use diverse, legally licensed, naturally authored content; deduplicate and track provenance.
2. Hold out entirely independent books/documents/topics, not only other numerical variations of the same templates.
3. Report actual tokenizer-measured token counts and loss curves; inspect both training and validation failure modes.
4. Train longer contexts on genuinely long documents; merely setting context_length to 6144 is not enough.
5. Apply task-specific LoRA fine-tuning only after completing base pretraining; evaluate on human-written prompts.
6. Avoid false claims that any language model is incapable of memorizing or copying its training examples.
