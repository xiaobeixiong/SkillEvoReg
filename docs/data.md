# Data preparation

Raw workbook, document and PDF attachments are distributed separately.
The bundled `data/manifests/` files record task membership, ordering and hashes.

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| SpreadsheetBench | 100 | 100 | 200 |
| SearchQA | 400 | 200 | 1,400 |
| LiveMath | 100 | 80 | 217 |
| Continual, each domain | 1--50 | — | 51--100 |

SkillEvolBench task definitions and native skill assets are included under
`benchmark/third_party/SkillEvolBench/benchmark/`. Learning anchors are generated
by the commands in [reproduction.md](reproduction.md); pretrained run snapshots
are not bundled.

To copy prepared data from a checkout with the expected directory layout:

```bash
python scripts/stage_data.py --from-project /path/to/checkout --dry-run
python scripts/stage_data.py --from-project /path/to/checkout
```

Staging copies declared datasets and Continual train/holdout task definitions.
It does not overwrite differing files or copy trajectories or credentials.
Local data paths are remapped to the destination checkout.

For Continual, build the staged Dockerfiles and image tags referenced in
`task.toml`. Build dependency images before domain train/holdout images.
The Dockerfiles include the native create-skill and modify-skill helpers.
Embedding weights are supplied separately using
`SKILLEVOREG_EMBEDDING_MODEL_PATH`.
