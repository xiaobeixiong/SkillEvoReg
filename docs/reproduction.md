# Running experiments

Run commands from the repository root. Set `OPENAI_API_KEY`,
`OPENAI_BASE_URL` and, for local embedding retrieval,
`SKILLEVOREG_EMBEDDING_MODEL_PATH` as shown in `.env.example`.
Install the corresponding requirements in a separate environment for each host.
The launcher selects the appropriate bundled Harbor runtime.

## SkillOpt

SpreadsheetBench, SearchQA and LiveMath each provide baseline and SkillEvoReg:

```bash
python scripts/run.py skillopt spreadsheet baseline --execute
python scripts/run.py skillopt spreadsheet skillevoreg --execute
python scripts/run.py skillopt searchqa baseline --execute
python scripts/run.py skillopt searchqa skillevoreg --execute
python scripts/run.py skillopt livemath baseline --execute
python scripts/run.py skillopt livemath skillevoreg --execute
```

Use `--cfg-options` for native configuration overrides. The native trainer
stores checkpoints and runtime state under the configured output root.
Do not share output directories between methods.

For frozen evaluation, add
`--evaluate --skill /path/to/skill.md --split test`.
Each invocation gets a new evaluation directory. To resume the same independent
observation, add `--eval-repeat 1`; another repeat index selects another directory.
`--eval-out-root /path/to/evaluation` selects an explicit output directory.
Existing results are reused only when the skill and evaluation identity match;
unmarked results and training directories are rejected. API credentials and the
base URL come from the environment; the distributed configs do not pin a provider.
The generic checkpoint-replicate helper is
`benchmark/scripts/run_reusable_checkpoint_replicates.py`; pass explicit
configuration, skill and training-root paths. Epoch-end delivery selects the
epoch with the highest recorded validation score, using the earlier epoch on ties.

## SkillEvolBench

Run a complete first-pass learning anchor, then export it:

```bash
python scripts/run.py evol anchor --run-id anchor --execute
PYTHONPATH=benchmark/third_party/SkillEvolBench python benchmark/third_party/SkillEvolBench/scripts/export_learning_anchor.py \
  --source-run benchmark/third_party/SkillEvolBench/workspace/runs/anchor \
  --output-dir benchmark/runtime/shared_anchor
```

The anchor contains all 90 first-pass T1--T3 learning records and per-environment
libraries. Keep the completed source run alongside its export. All three
conditions use this same first-pass state:

```bash
python scripts/run.py evol baseline --anchor benchmark/runtime/shared_anchor --repeat 1 --execute
python scripts/run.py evol twopass --anchor benchmark/runtime/shared_anchor --run-id twopass --execute
python scripts/run.py evol skillevoreg --anchor benchmark/runtime/shared_anchor --run-id skillevoreg --execute
```

The baseline command only freezes and evaluates the anchor's source libraries
on T4--T6 and T1--T3 replay (90 tasks per block). It does not train again.
Use `--repeat 2` for its second independent frozen evaluation. Both second-pass
commands load the same exported libraries; the launcher verifies that their
persistent contents match the first-pass source before starting.

All environments follow the native serial schedule. Frozen evaluation requires
an explicitly supplied completed run:

```bash
python scripts/evaluate_evol.py --source /path/to/completed_run --method regularization --block t4-t6 --execute
python scripts/evaluate_evol.py --source /path/to/completed_run --method regularization --block t1-t3 --execute
```

Use `baseline` or `noreg` for the other methods. The two blocks evaluate
T4--T6 and T1--T3 replay respectively. `--repeat` selects an independent
evaluation output label.

## ContinualSkillBench

Domains: `math`, `finance`, `law`, `healthcare`, `office`.

```bash
python scripts/run.py continual law baseline train --execute
python scripts/run.py continual law regularization train --execute
```

Export the step-50 libraries with
`benchmark/scripts/build_continual_<domain>_frozen_libraries.py`.
Math requires `--baseline-source` pointing to the step-50 state archive and
`--regularization-source` pointing to the step-50 final JSON. Other domains
require `--baseline-job` and `--regularization-job`.

Evaluate tasks 51--100 with the frozen step-50 library:

```bash
python scripts/run.py continual law baseline test --frozen-library /path/to/baseline_step050 --execute
python scripts/run.py continual law regularization test --frozen-library /path/to/regularization_step050 --execute
```

The frozen evaluator uses independent task sessions, skill-priority instructions,
immutable skills and the full-output judge. Math's configured learning sequence
contains 100 tasks; use its step-50 state for the 50/50 protocol. The other
domain training configurations contain tasks 1--50.

For two independent test observations, reuse the same frozen configuration and
library with a different `--run-name` for each observation; average their scores.

For checkpoint recovery, supply `--checkpoint /path/to/complete_checkpoint`
and a fresh `--run-name`. The preparation helper
`benchmark/scripts/prepare_continual_resume.py` additionally checks checkpoint
format and protocol identity.
