"""Step-local Regularization orchestration with explicit environment/model ports.

No default fake CCV, no live filesystem writes, and no hidden budget constants.
Runtime adapters supply measured complexity, matched windows, model and probes.
"""
from dataclasses import dataclass
from typing import Callable

from .policy import classify_changes, feedback_modules, render_instruction
from .transactions import apply_repair, paired_regression


@dataclass(frozen=True)
class Settings:
    create_complexity_budget: float
    modify_relative_growth: float
    create_atom_budget: int
    modify_atom_growth: int
    ccv_minimum_drop: float

    def __post_init__(self):
        if min(self.create_complexity_budget, self.modify_relative_growth,
               self.create_atom_budget, self.modify_atom_growth,
               self.ccv_minimum_drop) < 0:
            raise ValueError('Negative Regularization setting')


def skill_files(files):
    from .policy import META_SKILLS
    return {p: t for p, t in files.items() if p.endswith('/SKILL.md')
            and p.split('/')[0] not in META_SKILLS}


class StepRegularization:
    def __init__(self, settings: Settings, *, measure: Callable,
                 retrieve: Callable, local_edit: Callable, ccv: Callable,
                 repair_regression: Callable, measure_scope=None):
        self.settings = settings
        self.measure = measure
        self.retrieve = retrieve
        self.local_edit = local_edit
        self.ccv = ccv
        self.repair_regression = repair_regression
        self.measure_scope = measure_scope or (lambda files, path: self.measure(files.get(path, '')))

    def finish(self, old, restored_raw, *, masked_paths, source):
        """source is a permitted past/current successful training task, never future data.

        retrieve returns {history: [path], windows: {id: Window}}. Only the
        retrieval adapter can authorize windows. LLM output cannot expand them.
        CCV port owns clean qualification and isolated attacked evaluation;
        returned probe pair identities/scores are checked again here.
        """
        from .policy import META_SKILLS
        for path in old.keys() | restored_raw.keys():
            if path.split('/')[0] in META_SKILLS and old.get(path) != restored_raw.get(path):
                raise ValueError('Native updater changed a protected meta-skill')
        result = dict(restored_raw)
        changes = classify_changes(skill_files(old), skill_files(restored_raw))
        # An auxiliary-file edit also makes its owning skill responsible.
        for path in old.keys() | restored_raw.keys():
            skill = path.split('/')[0] + '/SKILL.md'
            if old.get(path) != restored_raw.get(path) and skill in skill_files(old) and skill in skill_files(restored_raw) and skill not in changes['modify']:
                changes['modify'].append(skill)
        changes['modify'].sort()
        events = []
        events.extend(dict(path=p, mode='delete', status='native_delete',
                           old_complexity=self.measure_scope(old, p)) for p in changes['delete'])
        # Fixed source list: repairs never recursively create more repair steps.
        for mode in ('create', 'modify'):
            for path in changes[mode]:
                if path not in result:
                    continue
                before = self.measure_scope(result, path)
                anchor = self.measure_scope(old, path)
                max_total = (self.settings.create_complexity_budget if mode == 'create'
                             else anchor['total'] * (1 + self.settings.modify_relative_growth))
                max_atoms = (self.settings.create_atom_budget if mode == 'create'
                             else anchor['atom_count'] + self.settings.modify_atom_growth)
                exceeded = before['total'] > max_total or before['atom_count'] > max_atoms
                modules = feedback_modules(dropout_consumed=bool(masked_paths),
                                           complexity_exceeded=exceeded)
                if not modules:
                    events.append(dict(path=path, mode=mode, status='not_triggered'))
                    continue
                matches = self.retrieve(mode=mode, path=path, old=dict(old), candidate=dict(result))
                history = set(matches['history'])
                if not history <= (skill_files(old).keys() & result.keys()) - {path}:
                    raise ValueError('Retrieval returned non-historical skill')
                request = dict(mode=mode, source_path=path, modules=modules,
                               instruction=render_instruction(mode, modules),
                               budget=dict(total=max_total, atoms=max_atoms),
                               candidate=dict(result), old=dict(old), matches=matches)
                receiving = {}
                for hp in sorted(history):
                    ha = self.measure_scope(old, hp)
                    receiving[hp] = dict(before=self.measure_scope(result, hp),
                        budget=dict(total=ha['total'] * (1 + self.settings.modify_relative_growth),
                                    atoms=ha['atom_count'] + self.settings.modify_atom_growth))
                request['receiving_skill_budgets'] = receiving
                response = self.local_edit(request)
                result = apply_repair(result, response, source=path, history=history,
                                      mode=mode, windows=matches.get('windows'))
                after = self.measure_scope(result, path)
                for hp, row in receiving.items():
                    row['after'] = self.measure_scope(result, hp)
                    row['still_over_budget'] = (row['after']['total'] > row['budget']['total'] or
                                               row['after']['atom_count'] > row['budget']['atoms'])
                events.append(dict(path=path, mode=mode, modules=list(modules),
                                   dropout_trigger='step_masked' if masked_paths else 'none',
                                   masked_paths=sorted(masked_paths), consumption_verified=False,
                                   receiving_skills=receiving,
                                   operation=response['operation'], before=before, after=after,
                                   still_over_budget=after['total'] > max_total or after['atom_count'] > max_atoms))
        # No extra compression pass and no budget rejection after Local editing.
        probe = self.ccv(old=dict(old), candidate=dict(result), source=source)
        status = probe['status']
        if status == 'paired':
            status = paired_regression(probe['old'], probe['new'],
                                       minimum_drop=self.settings.ccv_minimum_drop)
        elif status not in {'no_attack', 'inconclusive', 'clean_failed'}:
            raise ValueError('Unknown CCV outcome')
        if status == 'regression':
            responsibility = set(probe['responsible_paths'])
            if not responsibility or not responsibility <= (skill_files(result).keys() | skill_files(old).keys()):
                raise ValueError('Invalid CCV responsibility')
            probe = dict(probe, deleted_skill_context={p: old[p] for p in responsibility if p not in result})
            repaired = self.repair_regression(candidate=dict(result), probe=probe)
            modified = {p for p in result.keys() | repaired.keys() if result.get(p) != repaired.get(p)}
            if not modified <= responsibility:
                raise ValueError('CCV repair escaped responsibility')
            if any(p not in repaired or not repaired[p].strip() for p in responsibility):
                raise ValueError('CCV must preserve responsibility skills')
            result = dict(repaired)
        events.append(dict(module='ccv', status=status, regression_kind=probe.get('regression_kind'), reason=probe.get('reason'),
                           source_task_id=probe.get('source_task_id', source.get('task_id') if source else None),
                           source_policy=probe.get('source_policy', 'supplied_training_source'),
                           deleted_paths=changes['delete']))
        return result, events
