# MATEN/T MiniAI v1

**BY MATEN/T — Maten Faesal** · [M/ License v1.0](LICENSE.md)

MiniAI is a small, from-scratch decoder language model and an end-to-end training project. This repository includes the model architecture, streaming dataset generator, tokenizer preparation, training and evaluation utilities, checkpoint support, tests and a local chat interface. It **does not include pretrained model weights**.

## Architecture

| Component | Configuration |
| --- | --- |
| Parameters | **86,045,328** |
| Maximum context | **6,144 tokens (6K)** |
| Transformer | 10 layers, 12 attention heads, 768 hidden width |
| Feed-forward width | 3,240 |
| Positions | Rotary Position Embeddings (RoPE) |
| Tokenizer | SentencePiece BPE, 16,384 vocabulary pieces |
| Optimization | AdamW, FP16 AMP on supported CUDA devices, gradient accumulation, warmup and cosine LR decay |
| Training data | Up to 9,400,000 procedurally generated JSONL records from 94 domains |

The current version replaces the earlier 2K learned-position architecture. **Old 2K checkpoints are incompatible with the new 6K/RoPE model.** The maximum context is an architectural capacity, not proof of long-context accuracy.

## Dataset

The data generator covers 50 programming languages, 20 natural languages and 24 other subjects, including conversation, programming, mathematics, sciences and defensive security. The language datasets remain multilingual; English-language project documentation does not remove Sorani, Kurmanji or any other data domain.

Generated records are primarily **synthetic template variants**. Counting 9.4 million records does not mean 9.4 million unique expertly reviewed conversations. A complete training run alone cannot guarantee useful output.

## Quick setup

Install Python 3.10+ and the appropriate [PyTorch build for your hardware](https://pytorch.org/get-started/locally/), then:

```sh
python -m pip install -r requirements.txt
python brand_guard.py
python miniai.py info
```

Generate the complete dataset and train the tokenizer once:

```sh
python build_datasets.py --count 100000
python miniai.py prepare --samples-per-domain 1000
```

The GitHub source checkout contains neither the generated 9.4M records nor `tokenizer/tokenizer.model`. The preparation commands above are necessary before training a fresh clone. Ensure there is sufficient disk space.

## One-command full training

To generate the complete dataset, prepare the tokenizer and launch a full uncapped training epoch from one command, run:

```sh
python train_full.py
```

The launcher checks shard counts and SHA-256 hashes. If it detects a modified or partial dataset, it preserves the original and generates a fresh complete corpus in a separate folder. It resumes from a compatible `checkpoints/last.pt` if one exists. This can take substantial time and disk space; it runs only on a computer where you explicitly start it.

To select the full 6K training window on supported CUDA hardware with sufficient memory:

```sh
python train_full.py --device cuda --precision fp16 --sequence-length 6144
```

## Full training

This is the **full-epoch command**, not a 100-step demonstration. With the complete 9.4M-row dataset present, a 90% training split processes **8,460,000 records per epoch**. The exact token count depends on the tokenizer and content; it is logged as training proceeds.

```sh
python miniai.py train --device auto --precision auto --epochs 1 --sequence-length 1024 --batch-size 1 --grad-accum 8 --save-every 500
```

The default `--max-steps` value is `0`, so training proceeds until the requested epochs finish, unless you stop the process or the runtime fails. Streaming packing is enabled by default to make better use of short examples. This does not mean training runs automatically on GitHub.

For a CUDA GPU with sufficient memory, you can select the **full 6K training window**:

```sh
python miniai.py train --device cuda --precision fp16 --epochs 1 --sequence-length 6144 --batch-size 1 --grad-accum 8 --save-every 200
```

A 6K training sequence can be memory-intensive. Reducing `--sequence-length` to 1024 or 256 lowers training memory use without changing the model's maximum 6K context capacity. FP16 mixed precision does not make a randomly initialized model more knowledgeable on its own.

## Resume after interruption

Training saves a trusted checkpoint to `checkpoints/last.pt`. Restart with the same model, tokenizer, domains, precision, packing, learning-rate schedule, batch size and sequence length:

```sh
python miniai.py train --device auto --precision auto --epochs 1 --sequence-length 1024 --batch-size 1 --grad-accum 8 --save-every 500 --resume checkpoints/last.pt
```

To train an additional epoch after finishing the first, set `--epochs 2` and resume from the saved checkpoint. Interrupted runs resume from the latest saved microbatch, not necessarily the moment power was lost.

## Tests, evaluation and chat

```sh
python -m unittest discover -s tests -v
python miniai.py evaluate --checkpoint checkpoints/last.pt --split validation --max-batches 20
python miniai.py serve --device auto --port 8765
```

Open `http://127.0.0.1:8765` on the computer running the server. The local interface requires a trained checkpoint. Generated answers must be checked rather than assumed correct.

A tiny pipeline smoke test is available separately and **does not train the full model**:

```sh
python miniai.py train --tiny --sequence-length 64 --limit-samples 256 --max-steps 2 --output checkpoints-smoke
```

## Project documentation

- [Complete training guide for PC and Colab](training.txt)
- [Colab training notebook](notebooks/MATEN-T-MiniAI-v1-Colab.ipynb)
- [Model card](docs/MODEL_CARD.md)
- [Dataset card](docs/DATASET_CARD.md)
- [Quality and testing report](docs/QUALITY_REPORT.md)
- [M/ License](LICENSE.md)

No pretrained weights, professional benchmarks or completed full-dataset training run are included. This is a reproducible training project, not a ready-to-deploy production assistant. The attribution guard verifies required MATEN/T notices in standard entrypoints, but it is not tamper-proof DRM. M/ License is a custom license and is not OSI-approved.

**BY MATEN/T.**
