# 94 JSONL training-data shards — BY MATEN/T

Each `*.jsonl` file has **100,000 UTF-8 rows**, verified by line counting and SHA-256. See `manifest.json` and `../docs/DATASET_CARD.md`.

**Warning:** these are synthetic template variations, not 9.4 million independent expert-written texts; additional quality work is required. To regenerate datasets reproducibly:

```sh
python build_datasets.py --count 100000 --force
```

Do not assume rights to any external dataset merged into this directory. Only this project's owned material is offered under the custom M/ License v1.0.
