# MATEN/T MiniAI v1.1 — 100M, 6K, Pretraining + LoRA

**BY MATEN/T — Maten Faesal** · [M/ License v1.0](LICENSE.md)

A compact original PyTorch decoder language-model project. Version **1.1.0** introduces causal **next-token pretraining**, optional **LoRA supervised fine-tuning**, 6K-context RoPE attention and portable **SafeTensors export**. Source code is provided; **trained weights are not shipped**.

## Model specification

| Component | Specification |
| --- | --- |
| Trainable base parameters | **100,062,768** (~100M) |
| Context capacity | **6,144 tokens** |
| Vocabulary | **16,384** SentencePiece BPE pieces |
| Architecture | 10 Transformer blocks, hidden dimension 768, 12 attention heads, FFN 4,152, tied embeddings |
| Position encoding | Rotary Position Embeddings (RoPE) |
| Hardware precision | FP16 AMP + GradScaler on supported NVIDIA CUDA; BF16 where supported; FP32 fallback |
| Workflow | Full causal pretraining → optional LoRA chat fine-tuning → SafeTensors release |

**Version 1.1 is architecturally incompatible with all previous 86M checkpoints.** This release increases the FFN width and changes the parameter count. Retrain the model; do not attempt to resume an old checkpoint.

## What this project actually trains

**Pretraining** uses a causal next-token loss across **all** tokens in licensed text/JSONL documents, rather than only supervising the assistant's response. Text is streamed and packed into next-token training blocks; an end-of-document token separates documents. This teaches statistical structure and representations, but **cannot guarantee conceptual understanding or prevent memorization**.

**LoRA** is a separate fine-tuning stage that freezes the pretrained base model and learns lightweight attention adapters from instruction/response JSONL. It must run **after** a genuine pretrained checkpoint exists. The adapters can be merged for inference.

The optional generator builds **9,400,000 synthetic rows** across 94 domains (50 programming languages, 20 human languages and 24 conversation/science topics). These are repeated procedural templates, not 9.4 million independently reviewed documents. Good performance requires substantially more diverse, licensed and evaluated real-world text. The user must obtain consent/licenses for any custom corpus; never paste secret/private data without authorization.

## Install

Use Python 3.10+ and a supported PyTorch build for your operating system/GPU: [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

```bash
python -m pip install -r requirements.txt
python brand_guard.py
python -m unittest discover -s tests -v
```

A supported CUDA GPU is strongly recommended. CPU-only training can work, but full training of ~100M parameters on millions of examples may be extremely slow. A 6K training window can require substantially more VRAM than the default 512-token sequence.

## One-command full pretraining

```bash
python train_full.py
```

This **explicitly started command** prepares the 94 synthetic JSONL shards (if missing), trains a tokenizer, runs **one complete uncapped pretraining epoch**, saves resumable checkpoints and automatically writes an inference-ready release to `pretrain_checkpoints/release/`. There is **no 100-step limit** in the production command. If a compatible `last.pt` exists, the launcher resumes it. Do not assume the training job runs automatically on GitHub.

For a compatible CUDA GPU:

```bash
python train_full.py --device cuda --precision fp16 --sequence-length 512
```

For the full architectural window, **only if memory allows**:

```bash
python train_full.py --device cuda --precision fp16 --sequence-length 6144
```

To train on licensed external `.txt` or `.jsonl` files (recommended over relying exclusively on synthetic templates):

```bash
python train_full.py --corpus /path/to/licensed_corpus --tokenizer /path/to/tokenizer.model --device cuda --precision fp16
```

Missing tokenizers are trained from the **train split only** using `prepare_tokenizer.py`. To resume reliably, preserve corpus paths/content, tokenizer, hyperparameters and checkpoints.

## Optional LoRA after pretraining

After the base has been pretrained, run LoRA on properly licensed conversational JSONL:

```bash
python finetune_lora.py --base pretrain_checkpoints/last.pt --data datasets --device cuda --precision fp16
```

This saves `lora_output/adapter_model.safetensors` and `adapter_config.json`. A completed LoRA epoch also creates `lora_output/release_merged/model.safetensors` and associated configuration/tokenizer files. Use `--resume` to continue an interrupted adapter run. LoRA **does not replace** base pretraining.

For the single-command synthetic workflow with optional LoRA afterward:

```bash
python train_full.py --lora-after
```

## SafeTensors export and local chat

After a **completed pretraining run**, the exporter creates:

```text
pretrain_checkpoints/release/
  model.safetensors
  model.sha256
  config.json
  tokenizer.model
  tokenizer.vocab
  tokenizer_config.json
  special_tokens_map.json
  generation_config.json
  README.md
```

The output is a **custom PyTorch model format**, not a drop-in Transformers AutoModel checkpoint. Exported weights use FP16 by default and do not contain optimizer state. The full resumable state remains in `pretrain_checkpoints/last.pt`.

Manual export of a **trusted** training checkpoint:

```bash
python export_model.py --checkpoint pretrain_checkpoints/last.pt --output release-v1.1
```

Run local terminal chat or the browser interface from a release folder:

```bash
python chat_export.py --release pretrain_checkpoints/release
python chat_export.py --release pretrain_checkpoints/release --serve
```

Open `http://127.0.0.1:8765` on the computer running the server. Chat ability is **not guaranteed** by pretraining alone; instruction tuning and human evaluation matter.

## Training controls and verification

The default `pretrain.py --max-steps` is **0** (no artificial cap). Use `--max-steps` and `--tiny` **only for isolated pipeline tests**. The trainer reports actual target token counts, tracks optimizer steps and stores them in atomic checkpoints. An earlier ~343M-token figure was an estimate, **not** a verified tokenizer census.

The dataset generator also supports selecting a subset of domains and the tokenizer can be trained on custom documents. Exact duplicate filtering uses a bounded configurable window; it is **not** global deduplication or protection against memorization. Hold out genuinely independent documents and audit all training sources.

See [training.txt](training.txt), [Model card](docs/MODEL_CARD.md), [Dataset card](docs/DATASET_CARD.md) and [Colab notebook](notebooks/MATEN-T-MiniAI-v1-Colab.ipynb).

## Attribution and transparency

Project-authored code is published under the custom [M/ License v1.0](LICENSE.md), which is **not OSI-approved**. Third-party dependencies retain their own licenses. `brand_guard.py` checks for the project's original credit in standard code paths, but open source cannot provide unbreakable attribution enforcement. This repository is a research/training project, **not a fully trained assistant or a benchmarked production model**.

**BY MATEN/T.**
