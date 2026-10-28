"""Train-only ALFWorld CCV: paired, state-preserving action-menu permutation."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from skillopt.regularization.ccv import CCVResult
from .probe import ATTACK_TYPES

PROMPT = '''Audit update-induced dependence on incidental presentation features.
Choose at most ONE attack_type supported by changed skill text:
- admissible_action_order: deterministically permute the displayed action list; test positional priority shortcuts.
- action_menu_format: render exactly the same commands as a JSON string array instead of single-quoted lines; test quote/list-format assumptions.
- observation_layout: collapse observation whitespace to single spaces, preserving all words, identifiers, punctuation and order; test line/paragraph layout assumptions.
Do not propose object renaming, location swaps, goal paraphrases or other unsupported transformations.
The controller preserves game state, goal, action multiset, transitions and verifier.
Do not invent weaknesses, alter task goals, inject instructions, or use held-out data.
If the skill delta does not support a falsifiable hypothesis for these attacks, return no_attack.
Return JSON: {"decision":"attack"|"no_attack", "attack_type":"one supported type", "source_id":"supplied ID",
"delta_evidence":"exact changed text", "predicted_failure":"specific behavior",
"transferable_invariant":"an actionable execution rule preserving task-relevant behavior, not advice about testing or attacking; no example-specific details"}.
An attack proposal is NOT evidence of regression; only paired simulation decides it.'''


def propose(payload, out_dir, model, timeout, effort):
    from skillopt.model.codex_harness import prepare_workspace, run_regularization_codex_exec
    work = Path(out_dir) / 'attacker'
    prepare_workspace(work_dir=str(work), skill_md='', task_text=PROMPT + '\n' + json.dumps(payload))
    run_regularization_codex_exec(work_dir=str(work), prompt='Read task.md. Write JSON only to proposal.json. Stay inside this workspace.',
                         model=model, timeout=timeout, reasoning_effort=effort)
    return json.loads((work / 'proposal.json').read_text())


def run_alfworld_ccv(adapter, *, old_skill, candidate_skill, seed, out_dir,
                      trajectory_context=None, source_split='train', source_size=1,
                      attacker_model='gpt-5.4-mini', agent_timeout=1200,
                      reasoning_effort='medium', provider='codex_exec', **kwargs):
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = CCVResult(status='no_attack')
    def finish():
        data = result.to_dict()
        tmp = root / 'ccv_result.json.tmp'
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(root / 'ccv_result.json')
        return data
    if source_split != 'train':
        raise ValueError('ALFWorld CCV sources must be train-only')
    if old_skill == candidate_skill:
        return finish()
    if provider != 'codex_exec':
        raise ValueError('ALFWorld CCV currently supports isolated codex_exec only')
    from skillopt.envs.alfworld.adapter import ALFWorldBatchRun
    train = {str(x['id']): x for x in adapter.dataloader.train_items}
    # Strict allowlist: never forward selection results, prior records or unrelated trajectories.
    successful = {}
    for row in (trajectory_context or {}).get('current_step_rollouts', []):
        key = str(row.get('id', ''))
        if key in train and row.get('agent_ok') is True and row.get('hard') == 1:
            successful[key] = {'id': key, 'task_description': row.get('task_description', ''),
                               'task_type': train[key].get('task_type', '')}
    sources = list(successful.values())[:max(0, source_size)]
    if not sources:
        return finish()
    try:
        proposal = propose({'old_skill': old_skill, 'candidate_skill': candidate_skill, 'sources': sources},
                           out_dir, attacker_model, agent_timeout, reasoning_effort)
        if proposal.get('decision') == 'no_attack':
            return finish()
        key = str(proposal.get('source_id', ''))
        attack_type = proposal.get('attack_type', 'admissible_action_order')
        if attack_type not in ATTACK_TYPES:
            raise ValueError('unsupported CCV attack type')
        if proposal.get('decision') != 'attack' or key not in {x['id'] for x in sources}:
            raise ValueError('invalid CCV decision or source')
        evidence = str(proposal.get('delta_evidence', '')).strip()
        if not evidence or evidence not in candidate_skill or evidence in old_skill:
            raise ValueError('CCV delta evidence is not candidate-specific')
        if not str(proposal.get('predicted_failure', '')).strip():
            raise ValueError('missing falsifiable prediction')
        result = replace(result, generated_attacks=1, materialized_attacks=1, validator_passed_attacks=1)
        item = train[key]
        batch = ALFWorldBatchRun(env_num=1, eval_dataset='train', seed=seed, is_train=True,
                                 workers=1, specific_gamefiles=[item['gamefile']], result_ids=[key], items=[item])
        scores = {}
        for slot, skill, attack in [('old_clean', old_skill, False), ('candidate_clean', candidate_skill, False),
                                    ('old_attack', old_skill, True), ('candidate_attack', candidate_skill, True)]:
            # Do not reuse masked train success as proof that the unmasked old skill passed.
            if attack and not all(scores.get(s) == 1 for s in ['old_clean', 'candidate_clean']):
                break
            probe = replace(batch, ccv_order_seed=seed if attack else None,
                            ccv_attack_type=attack_type if attack else None)
            rows = adapter.rollout(probe, skill, str(root / slot))
            if len(rows) != 1 or rows[0].get('agent_ok') is not True or rows[0].get('hard') not in (0, 1):
                raise RuntimeError('unscoreable paired probe')
            scores[slot] = rows[0]['hard']
            if attack:
                evidence_rows = rows[0].get('ccv_presentation_evidence', [])
                if not evidence_rows or not any(e.get('changed') for e in evidence_rows):
                    raise ValueError('probe produced no verified presentation change')
                if not all(e.get('action_multiset_preserved') and e.get('observation_tokens_preserved') for e in evidence_rows):
                    raise ValueError('probe presentation invariants failed')
        qualified = scores.get('old_clean') == scores.get('candidate_clean') == 1
        paired = qualified and 'candidate_attack' in scores
        regression = paired and scores['old_attack'] == 1 and scores['candidate_attack'] == 0
        record = {'source_id': key, 'attack_type': attack_type, 'order_seed': seed,
                  'attack_id': attack_type + '_1',
                  'status': 'update_regression' if regression else 'no_regression' if paired else 'inconclusive',
                  'pattern': {'incidental_assumption': proposal['predicted_failure'], 'delta_evidence': evidence},
                  'generation': {'intervention': attack_type + ': presentation only; preserve game, observation tokens and actions.'},
                  'scores': scores, 'confirmed_regression': regression, 'proposal': proposal,
                  'old_skill_sha256': hashlib.sha256(old_skill.encode()).hexdigest(),
                  'candidate_skill_sha256': hashlib.sha256(candidate_skill.encode()).hexdigest()}
        # Retain full evidence. No free-form claimed regression enters counts.
        result = replace(result, status='ok' if paired else 'inconclusive',
                         qualified_sources=int(qualified), source_qualification_failures=int(not qualified),
                         paired_evaluated_attacks=int(paired), valid_attacks=int(paired),
                         old_qualified_attacks=int(paired and scores['old_attack'] == 1),
                         old_successes=int(paired and scores['old_attack'] == 1),
                         candidate_successes=int(paired and scores['candidate_attack'] == 1),
                         confirmed_regressions=int(regression), regressions=int(regression),
                         patterns=1, attack_records=[record])
    except Exception as exc:
        result = replace(result, status='error', inconclusive_failures=1, error=f'{type(exc).__name__}: {exc}')
    return finish()
