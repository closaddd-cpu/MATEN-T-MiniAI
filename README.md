# MATEN/T MiniAI v1 — 86M · 2K · Multilingual

**BY MATEN/T — Maten Faesal** · [**M/ License v1.0**](LICENSE.md)

From-scratch educational decoder language model: **86,019,712 parameters**, **2,048-token context**, **16,384-token SentencePiece tokenizer**, PyTorch training, an optional C++17 data utility, a minimal dark/light local chat interface, and **94 UTF-8 JSONL files with 100,000 records each**.

**No `.bat`, API key, subscription, third-party model weights or private dataset is required.** Python 3.10+ is needed. The repo contains **training code and untrained model architecture**, **not trained 86M weights**.

> **Critical distinction:** 9.4 million *synthetic* records ≠ 9.4 million independent, expertly written examples. Many records are mathematically varied versions of a small set of templates. The project is a reproducible research/training starter kit, **not** a capable ready-to-use general-purpose assistant. Neither 50 programming shards nor 20 language shards establish competence in those languages. Diverse licensed material, native-speaker review, security review, independent benchmarks, and substantial GPU training are necessary before claiming real-world quality.

## Architecture

| Component | Value |
|---|---|
| Trainable parameters | **86,019,712** (counted from PyTorch model) |
| Model | 10-layer causal decoder Transformer, 12 heads, width 768, FFN 3136 |
| Maximum context | **2,048 tokens** |
| Tokenizer | SentencePiece BPE, **16,384** vocabulary items |
| Training | AdamW, gradient accumulation, optional CUDA/ROCm-compatible or MPS backend if supported by installed PyTorch, CPU fallback |
| Inference | Local Python HTTP chat; model must be trained first |
| Optional C++ | Data row-count utility; model and optimization use PyTorch |
| Data | 94 files × 100,000 records = **9,400,000** synthetic rows |

`--sequence-length 256` is a lower-memory training setting; it does **not** change the model's 2,048-token capacity. To train on 2K spans specify `--sequence-length 2048` only if memory allows.

## Dataset layout

- **50 programming languages:** `c`, `c++`, `java`, `python`, `javascript`, `typescript`, `csharp`, `go`, `rust`, `kotlin`, `swift`, `php`, `ruby`, `bash`, `powershell`, `sql`, `lua`, `dart`, `r`, `julia`, `scala`, `groovy`, `perl`, `elixir`, `erlang`, `haskell`, `ocaml`, `fsharp`, `clojure`, `common_lisp`, `scheme`, `racket`, `fortran`, `pascal`, `delphi`, `zig`, `nim`, `crystal`, `d`, `objective_c`, `matlab`, `octave`, `visual_basic`, `awk`, `tcl`, `gdscript`, `coffeescript`, `solidity`, `ada`, `prolog`.
- **20 natural languages:** `kurdish_sorani`, `kurdish_kurmanji`, `english`, `arabic`, `turkish`, `persian`, `french`, `german`, `spanish`, `italian`, `portuguese`, `russian`, `hindi`, `urdu`, `chinese_simplified`, `japanese`, `korean`, `dutch`, `swedish`, `indonesian`.
- **24 other subjects and dialog:** `conversation_human`, `conversation`, `cybersecurity`, `mathematics`, `physics`, `chemistry`, `biology`, `astronomy`, `ai_foundations`, `computer_science`, `statistics`, `geology`, `geography`, `ecology`, `engineering`, `electronics`, `robotics`, `data_science`, `environmental_science`, `economics`, `health_literacy`, `linguistics`, `logic`, `cryptography`.

Each JSONL line is valid JSON and carries: `id`, `domain`, `topic`, `instruction`, `response`, `created_by`, `license`, `provenance`, `review_status`. See `datasets/manifest.json` for SHA-256 checksums and [DATASET_CARD](docs/DATASET_CARD.md) for quality limitations and license scope. 100K per file does not imply 100K distinct topics; conversational datasets also use math-based variants.

## Installation (PC, Linux, macOS)

```sh
python -m pip install -r requirements.txt
python miniai.py info
python -m unittest discover -s tests -v
```

CUDA is supported only where PyTorch and appropriate drivers support it. The app checks for CUDA, then MPS, then CPU. It does not promise acceleration for every GPU. No specific GPU hardware is named or required.

## Create data and retrain tokenizer

The **complete package includes the generated JSONL data and tokenizer**. If using the **code-only** archive, first generate the shards yourself:

```sh
python build_datasets.py --count 100000
python miniai.py prepare --samples-per-domain 1000
```

The dataset generator is streaming and deterministic and writes ~4 GB of plain JSONL for the full catalogue; leave sufficient free disk space. To create selected shards:

```sh
python build_datasets.py --count 100000 --domains kurdish_sorani arabic python cybersecurity
```

`prepare` learns tokenizer subwords only from train-set rows. **Do not swap tokenizer after you start training** because old checkpoints will no longer be compatible. The provided bundled tokenizer is for this 94-shard corpus.

## Train from random initialization

```sh
python miniai.py train --device auto --epochs 1 --batch-size 1 --grad-accum 8 --sequence-length 256 --save-every 500
```

Train at full maximum sequence length **only if your hardware allows**:

```sh
python miniai.py train --device cuda --epochs 1 --batch-size 1 --grad-accum 16 --sequence-length 2048 --save-every 200
```

Resume an existing trusted checkpoint with exactly matching model/tokenizer/data/hyperparameters:

```sh
python miniai.py train --device auto --epochs 1 --batch-size 1 --grad-accum 8 --sequence-length 256 --resume checkpoints/last.pt
```

Short test (smaller model, **not** 86M):

```sh
python miniai.py train --tiny --sequence-length 64 --limit-samples 188 --batch-size 2 --grad-accum 2 --max-steps 2 --output checkpoints-smoke
```

`--limit-samples` is round-robin across all shards; without it training streams all selected data. A complete epoch through 9.4M rows may take **very long** and could outlast a Colab session. Save checkpoints to persistent storage. Checkpoint files may be hundreds of MB or more. No guarantee of convergence or high-quality output.

## Validate dataset integrity

```sh
python tools/verify_datasets.py
```

For a trusted checkpoint, a validation loss check is also available:

```sh
python miniai.py evaluate --checkpoint checkpoints/last.pt --split validation --max-batches 20
```

## Local chat interface

**After training** creates `checkpoints/last.pt`:

```sh
python miniai.py serve --device auto --port 8765
```

Open <http://127.0.0.1:8765>. The Chat UI has dark/light mode, responsive layout, emojis and 20-language answer preference. **The language selector adds a prompt hint; it does not guarantee that a small or poorly trained model is fluent in that language.** The HTTP server binds only to localhost; it is not intended for public deployment.

## Colab

Open [`notebooks/MATEN-T-MiniAI-v1-Colab.ipynb`](notebooks/MATEN-T-MiniAI-v1-Colab.ipynb) and follow the cells; upload the full ZIP to Google Drive and change ZIP_PATH if needed. Enable GPU runtime and monitor progress. A 9.4M-row full epoch can require far more time than one free Colab runtime. Model weights are saved to Drive.

## License, verification and transparency

Project-authored code and generated examples are distributed under the custom [M/ License v1.0](LICENSE.md). This license is **not OSI-approved** and has **not had a legal review**. Dependencies such as Python/PyTorch/SentencePiece have their **own licenses**, and third-party contributions must be independently licensed. See [QUALITY_REPORT](docs/QUALITY_REPORT.md) and [MODEL_CARD](docs/MODEL_CARD.md).

This project covers educational, defensive cybersecurity data only. Do not treat the generated text as professional medical or security advice. **BY MATEN/T.**

## MATEN/T attribution guard

The source release includes `brand_guard.py`. `python miniai.py ...` and
`python build_datasets.py ...` validate the presence of the original MATEN/T
attribution in the key code, README, license and chat-UI files before starting.
If a required notice is removed, the command stops with an explicit
`M/ License` message. Run `python brand_guard.py` to validate the notices.

**Limitations:** This is a local consistency check, not unbreakable DRM. A user
with complete source access can bypass or modify it. The guard does not delete
anything, modify dataset rows or checkpoints, or make network calls. The license
text itself, not this guard, describes redistribution obligations. See
`docs/ATTRIBUTION.md` for detailed behavior and tests.


## GitHub source distribution

This repository publishes the **source code and the dataset generator**, not the 9.4M-row generated data, tokenizer binary, or trained checkpoints. To produce data locally, run `python build_datasets.py` and follow the preparation instructions above before model training. Source code is not a trained AI model.
