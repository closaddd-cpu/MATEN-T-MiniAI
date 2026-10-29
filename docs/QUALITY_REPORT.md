# MATEN/T MiniAI v1 — quality and limitations

**BY MATEN/T · M/ License v1.0**

## Verified in this project build

- Dataset manifest lists **94 shards × 100,000 = 9,400,000** records.
- Independent streaming **SHA-256 and line-count verification across all 94 files succeeded**: 0 mismatches. This checks byte integrity and metadata on first records, **not factual quality** of all rows.
- **11/11 Python unit tests passed** including model gradient, configuration, tokenizer scripts, category catalog, example formatting and manifest.
- The current 86M architecture has 10 layers and 6K RoPE context. The full 86M model is **not trained in this release**.
- A **2,444,544-parameter tiny smoke model** performed forward/backward, two optimizer steps and saved a checkpoint. A validation evaluation call completed, but its numeric loss does **not** measure usefulness of the untrained 86M model.
- Sample snippets from **C, C++, Python, JavaScript, Ruby, Perl, PHP, and AWK** were executed and checked against generated expected outputs. The remaining programming language samples were **not exhaustively compiled or run**.
- SentencePiece BPE retrained from **94,000 train-set examples** sampled from all 94 shards, with 16,384-vocabulary configuration.

## Not verified / cannot honestly promise

- Per-record subject matter correctness, native translation quality, expert scientific depth, humanlike conversation, factual accuracy, security advice quality, robust code generation, or GPU portability across all hardware.
- Full 86M training convergence on 9.4M rows or completion within any particular PC/Colab session.
- True independent validation: training and validation share template families even when row IDs differ.
- The full set of programming languages in existence: this catalogue intentionally includes **50 named languages** and is extensible, but is not a literal exhaustive list.

## Recommended quality upgrades

1. Replace repetitive math-variant samples with more independently authored consented/licensed dialogues and reviewed, tested code.
2. Recruit native speakers for Sorani, Kurmanji, Arabic and each remaining language; review style and correctness.
3. Increase scientific task diversity and create expert-reviewed advanced problem sets instead of claiming depth from row counts.
4. Evaluate on out-of-template, cross-domain and human-authored test sets, and publish failures.
5. Keep a separate license/attribution inventory for any future third-party materials.

## October 2026 source update

The revised model has **86,045,328 parameters**, **6,144-token maximum context** and Rotary Position Embeddings. The training implementation supports dense packing, measured input/supervised token counts and full-epoch training without a default step cap. Historical dataset validation and smoke-test results above refer to prior builds; they are **not** independent verification of a completed 6K model training run. The new model is not checkpoint-compatible with the previous 2K learned-position version.
