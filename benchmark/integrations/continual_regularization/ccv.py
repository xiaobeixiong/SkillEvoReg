"""Math CCV: original-label transformations or independently checked boundary probes.

Boundary probes use a fresh numeric oracle, never the original task's grader.
Independent solving plus audit is not a formal mathematical proof.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor

from .baseline import DIRECT_PREFIX, codex_route_flags

# Application budget, not a claim about the provider's context-window size.
ATTACK_INPUT_TOKEN_BUDGET = 64000


def attack_budget_check(request):
    import tiktoken
    tokens = len(tiktoken.get_encoding('o200k_base').encode(
        json.dumps(request, ensure_ascii=False), disallowed_special=()))
    if tokens > ATTACK_INPUT_TOKEN_BUDGET:
        return dict(status='no_attack', reason='input_budget_exceeded',
                    input_tokens=tokens, input_token_budget=ATTACK_INPUT_TOKEN_BUDGET)
    return None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def parse_codex_event_stream(stdout):
    """Parse lifecycle evidence without letting a malformed tool event abort CCV.

    Codex can occasionally emit an invalid JSON line for a very large intermediate
    command result while still ending the turn with a valid ``turn.completed``.
    Such an item is diagnostic noise, not evidence that the solve failed.  A
    malformed lifecycle event remains fail-closed.
    """
    events, malformed = [], []
    for line_number, line in enumerate(stdout.splitlines(), 1):
        if not line.startswith('{'):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            malformed.append({
                'line_number': line_number,
                'error': str(exc),
                'lifecycle': any(marker in line for marker in (
                    '"type":"turn.completed"', '"type":"turn.failed"')),
                'sha256': hashlib.sha256(line.encode('utf-8', 'replace')).hexdigest(),
                'preview': line[:1000],
            })
    completed = any(event.get('type') == 'turn.completed' for event in events)
    failed = any(event.get('type') == 'turn.failed' for event in events)
    malformed_lifecycle = any(row['lifecycle'] for row in malformed)
    return events, malformed, completed and not failed and not malformed_lifecycle


class MathCCV:
    image = 'agent-overfit-continual-math:baseline-v1'
    attacks_module = 'continual_regularization.attacks'

    def __init__(self, model, directory, timeout, minimum_drop=0.1):
        self.model, self.directory, self.timeout = model, Path(directory), timeout
        self.minimum_drop = minimum_drop
        self.directory.mkdir(parents=True, exist_ok=True)

    def attacks(self):
        return importlib.import_module(self.attacks_module)

    @staticmethod
    def paired_slots(slot, old, candidate, instruction, source, prefix):
        # Preserve slot identities and the clean -> attack dependency. Only the
        # two independent libraries within one phase run concurrently.
        with ThreadPoolExecutor(max_workers=2) as pool:
            left = pool.submit(slot, old, instruction, source, prefix + '_old')
            right = pool.submit(slot, candidate, instruction, source, prefix + '_new')
            return left.result(), right.result()

    def clean_slot(self, files, instruction, source, name):
        # Run-local cache only. Pin actual image and grader/test implementation,
        # not merely the mutable image tag or a historical training pass.
        from harbor.agents.installed.sequential_subtask_judge import format_sequential_subtask_judge_script
        image = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}',
            self.image], capture_output=True, text=True,
            check=True, timeout=30).stdout.strip()
        if not image:
            raise RuntimeError('Cannot establish probe image identity')
        tests = Path(source['tests'])
        identity = dict(library=files, instruction=instruction, grading=source['grading'],
            task_id=source['task_id'], image=image, endpoint=os.environ['OPENAI_BASE_URL'],
            model='gpt-5.4-mini', effort='medium', timeout=self.timeout,
            harness=Path(__file__).read_text(),
            judge=format_sequential_subtask_judge_script(judge_model='openai/gpt-5.4-mini', pass_threshold=.6),
            tests={p.relative_to(tests).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(tests.rglob('*')) if p.is_file() and '__pycache__' not in p.parts})
        key = digest(identity)
        cache = self.directory.parent.parent / 'clean_cache'
        cache.mkdir(exist_ok=True)
        entry = cache / (key + '.json')
        if entry.is_file():
            data = json.loads(entry.read_text())
            observation = data['observation']
            if data['key'] != key or data['checksum'] != digest(observation):
                raise RuntimeError('Corrupted clean cache')
            if observation['status'] == 'ok' and observation.get('passed') is True:
                directory = self.directory / name
                directory.mkdir()
                (directory / 'observation.json').write_text(json.dumps(observation))
                (directory / 'reuse.json').write_text(json.dumps(dict(key=key, original=data['original'])))
                return observation
        observation = self.slot(files, instruction, source, name)
        if observation['status'] == 'ok' and observation.get('passed') is True:
            temporary = entry.with_suffix('.' + uuid.uuid4().hex + '.tmp')
            temporary.write_text(json.dumps(dict(key=key, observation=observation,
                checksum=digest(observation), original=str(self.directory / name))))
            temporary.replace(entry)
        return observation

    def slot(self, files, instruction, source, name):
        from harbor.agents.installed.sequential_subtask_judge import format_sequential_subtask_judge_script
        from .container_io import ROOTS
        directory = self.directory / name
        directory.mkdir()
        identity = dict(probe_hash=digest(instruction), grading_hash=digest(source['grading']),
                        library_hash=digest(files))
        container = 'continual-probe-' + uuid.uuid4().hex[:16]
        def run(args, timeout=60):
            return subprocess.run(['docker', *args], capture_output=True, text=True,
                                  timeout=timeout, check=True)
        try:
            run(['run', '-d', '--name', container, '--entrypoint', 'sleep',
                 self.image, 'infinity'])
            with tempfile.TemporaryDirectory(prefix='continual-probe-') as temporary:
                tmp = Path(temporary)
                library = tmp / 'skills'
                library.mkdir()
                for path, text in files.items():
                    if Path(path).is_absolute() or '..' in Path(path).parts:
                        raise ValueError('Invalid probe path')
                    target = library / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(text.encode('utf-8', 'surrogateescape'))
                for root in ROOTS:
                    run(['cp', str(library) + '/.', container + ':' + root])
                frozen_manifest = None
                if source.get('frozen_eval'):
                    manifest_command = 'find ' + ' '.join(shlex.quote(r) for r in ROOTS) + ' -type f -exec sha256sum {} + | sort'
                    frozen_manifest = run(['exec', container, 'bash', '-o', 'pipefail', '-c', manifest_command]).stdout
                    (directory / 'library_before.sha256').write_text(frozen_manifest)
                # Empty /app, no previous task answers, no task memory/session,
                # no grading payload in the solve process environment.
                prompt = ('Use relevant skills under /root/.agents/skills when helpful. '
                          'Solve the task and write its requested output. Do not update skills.\n' + instruction)
                command = DIRECT_PREFIX + 'codex exec ' + codex_route_flags() + (
                    ' --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox '
                    '--model gpt-5.4-mini -c model_reasoning_effort=medium --json -- ' + shlex.quote(prompt))
                response = run(['exec', '-e', 'OPENAI_API_KEY', '-e', 'OPENAI_BASE_URL',
                                container, 'bash', '-o', 'pipefail', '-c', command], timeout=self.timeout)
                (directory / 'codex.txt').write_text(response.stdout + response.stderr)
                events, malformed, completed = parse_codex_event_stream(response.stdout)
                if malformed:
                    (directory / 'malformed_events.json').write_text(
                        json.dumps(malformed, indent=2, ensure_ascii=False))
                if not completed:
                    raise RuntimeError('Probe execution incomplete')
                if frozen_manifest is not None:
                    after = run(['exec', container, 'bash', '-o', 'pipefail', '-c', manifest_command]).stdout
                    (directory / 'library_after.sha256').write_text(after)
                    if frozen_manifest != after:
                        raise RuntimeError('Frozen evaluation modified skill library')
                if source.get('boundary_oracle'):
                    # The expected answer is never copied into the solver container.
                    # Read only its answer after solve, and score on the host.
                    numeric_equal = self.attacks().numeric_equal
                    read = subprocess.run(['docker', 'exec', container, 'cat', '/app/scap_answer.json'],
                        capture_output=True, text=True, timeout=30)
                    (directory / 'answer.txt').write_text(read.stdout)
                    if read.returncode and 'No such file' not in read.stderr:
                        raise RuntimeError('Cannot read boundary answer: ' + read.stderr)
                    try:
                        answer = json.loads(read.stdout)
                    except json.JSONDecodeError:
                        answer = None
                    passed = isinstance(answer, dict) and numeric_equal(
                        answer.get('answer'), source['grading']['expected_answer'])
                    observation = dict(identity, status='ok', score=float(passed), passed=passed)
                    (directory / 'observation.json').write_text(json.dumps(observation, indent=2))
                    return observation
                judge = tmp / 'judge.py'
                judge.write_text(format_sequential_subtask_judge_script(
                    judge_model='openai/gpt-5.4-mini', pass_threshold=.6))
                run(['cp', str(judge), container + ':/root/judge_subtask.py'])
                run(['cp', source['tests'], container + ':/tests'])
                grading_env = dict(os.environ, HARBOR_GRADING_SPEC_JSON=json.dumps(source['grading']))
                command = DIRECT_PREFIX + f'python3 /root/judge_subtask.py --task-id {source["task_id"]} --output /tmp/probe-feedback.json'
                graded = subprocess.run(['docker', 'exec', '-e', 'OPENAI_API_KEY', '-e', 'OPENAI_BASE_URL',
                    '-e', 'OPENAI_API_BASE', '-e', 'HARBOR_GRADING_SPEC_JSON', container,
                    'bash', '-o', 'pipefail', '-c', command], env=grading_env,
                    capture_output=True, text=True, timeout=self.timeout, check=True)
                run(['cp', container + ':/tmp/probe-feedback.json', str(directory / 'feedback.json')])
                feedback = json.loads((directory / 'feedback.json').read_text())
                if feedback.get('judge_error'):
                    raise RuntimeError('Probe judge API error')
                score = float(feedback.get('score', feedback.get('total_score')))
                observation = dict(identity, status='ok', score=score, passed=score >= .6)
        except (subprocess.SubprocessError, RuntimeError) as exc:
            (directory / 'error.txt').write_text(str(exc))
            observation = dict(identity, status='error')
        finally:
            subprocess.run(['docker', 'rm', '-f', container], capture_output=True, timeout=30)
        (directory / 'observation.json').write_text(json.dumps(observation, indent=2))
        return observation

    def __call__(self, *, old, candidate, source):
        attacks = self.attacks()
        delta_context, select_source = attacks.delta_context, attacks.select_source
        delta = delta_context(old, candidate)
        if source and 'pool' in source:
            source, ranking = select_source(source['pool'], delta)
            (self.directory / 'source_selection.json').write_text(json.dumps(ranking, indent=2))
        (self.directory / 'delta.json').write_text(json.dumps(delta, indent=2))
        result = self._attack(old=old, candidate=candidate, source=source, delta=delta)
        result['source_task_id'] = source.get('task_id') if source else None
        result['source_policy'] = 'delta_embedding_top1_successful_training'
        (self.directory / 'summary.json').write_text(json.dumps(result, indent=2))
        return result

    def _attack(self, *, old, candidate, source, delta):
        attacks = self.attacks()
        GENERATE, VALIDATE, KINDS = attacks.GENERATE, attacks.VALIDATE, attacks.KINDS
        proposal_error, validated = attacks.proposal_error, attacks.validated
        SOLVE_BOUNDARY, VALIDATE_BOUNDARY = attacks.SOLVE_BOUNDARY, attacks.VALIDATE_BOUNDARY
        BOUNDARY_CHECKS = attacks.BOUNDARY_CHECKS
        numeric_equal, normalize_output_contract = attacks.numeric_equal, attacks.normalize_output_contract
        self.candidate = candidate
        if source is None:
            return {'status': 'no_attack', 'reason': 'no_successful_training_source'}
        if old == candidate:
            return {'status': 'no_attack', 'reason': 'identical_library'}
        request = dict(instruction=GENERATE, allowed_paths=sorted(delta), kinds=sorted(KINDS),
            source=source['instruction'], source_trajectory=source.get('trajectory', ''), delta=delta)
        over_budget = attack_budget_check(request)
        if over_budget:
            return over_budget
        clean_old, clean_new = self.paired_slots(
            self.clean_slot, old, candidate, source['instruction'], source, 'clean')
        if any(x['status'] != 'ok' for x in (clean_old, clean_new)):
            return {'status': 'inconclusive'}
        # Clean runs only qualify the source; repair is attack-regression only.
        if not clean_old['passed'] or not clean_new['passed']:
            return {'status': 'clean_failed'}
        request = dict(instruction=GENERATE, allowed_paths=sorted(delta), kinds=sorted(KINDS),
            source=source['instruction'], source_trajectory=source.get('trajectory', ''), delta=delta)
        for attempt in range(2):
            over_budget = attack_budget_check(request)
            if over_budget:
                return over_budget
            try:
                proposal = normalize_output_contract(self.model(request))
                error = proposal_error(proposal, delta, source)
            except json.JSONDecodeError:
                error = 'response was not a single JSON object'
                proposal = None
            if error is None:
                break
            (self.directory / f'invalid_proposal_{attempt+1}.json').write_text(json.dumps(dict(error=error, proposal=proposal)))
            request = dict(request, validation_error=error,
                           retry_instruction='Correct the schema and responsibility, or return attack=false. Do not invent blame.')
        else:
            return {'status': 'no_attack', 'reason': 'invalid_proposal'}
        if proposal['attack'] is False:
            return {'status': 'no_attack', 'reason': 'no_credible_weakness', 'detail': proposal.get('reason')}
        responsibility = proposal['responsible_paths']
        boundary = proposal['kind'] == 'boundary_counterexample'
        try:
            if boundary:
                independent = self.model(dict(instruction=SOLVE_BOUNDARY,
                    question=proposal['attacked_instruction']))
                (self.directory / 'independent_oracle.json').write_text(json.dumps(independent, indent=2))
                if (not isinstance(independent, dict) or independent.get('solvable') is not True
                        or not independent.get('derivation')
                        or not numeric_equal(independent.get('answer'), proposal['expected_answer'])):
                    return {'status': 'no_attack', 'reason': 'oracle_disagreement'}
                validation = self.model(dict(instruction=VALIDATE_BOUNDARY,
                    source=source['instruction'], proposal=proposal, independent_solution=independent))
                valid = isinstance(validation, dict) and validation.get('valid') is True and all(
                    validation.get(k) is True for k in BOUNDARY_CHECKS)
            else:
                validation = self.model(dict(instruction=VALIDATE, source=source['instruction'],
                    grading=source.get('grading'), proposal=proposal))
                valid = validated(validation)
        except json.JSONDecodeError:
            return {'status': 'no_attack', 'reason': 'invalid_validation'}
        if not valid:
            return {'status': 'no_attack', 'reason': 'invalid_probe'}
        attacked = proposal['attacked_instruction']
        (self.directory / 'attack.json').write_text(json.dumps(dict(proposal=proposal, validation=validation), indent=2))
        probe_source = (dict(source, boundary_oracle=True, grading=dict(
            type='boundary_numeric_v1', expected_answer=proposal['expected_answer'],
            absolute_tolerance=1e-9)) if boundary else source)
        attacked_old, attacked_new = self.paired_slots(
            self.slot, old, candidate, attacked, probe_source, 'attacked')
        return {'status': 'paired', 'regression_kind': 'attack_regression', 'old': attacked_old,
                'new': attacked_new,
                'responsible_paths': responsibility, 'proposal': proposal}

    def repair(self, *, candidate, probe):
        paths = probe['responsible_paths']
        request = {'instruction': 'A paired shortcut attack evaluation detected regression. '
            'Return {"skills":{path:full_text}} for the responsibility skills only. '
            'Repair the cause as a transferable invariant; prefer editing or merging the responsible rule. '
            'Preserve unrelated semantics. Do not add source answers, task IDs, or irrelevant cue details. '
            'No compression unless directly necessary for this repair. '
            'If a responsible skill was deleted, you may restore its necessary invariant using deleted_skill_context. '
            'Do not restore unrelated rules or claim missing auxiliary resources exist.',
            'probe': probe, 'skills': {p: candidate.get(p) for p in paths}}
        for attempt in range(3):
            try:
                reply = self.model(request)
                skills = self._repair_skills(reply, paths)
                return dict(candidate, **skills)
            except (ValueError, TypeError) as exc:
                # Schema/JSON failures only: transport exceptions must propagate.
                if attempt == 2:
                    raise ValueError('SCAP repair invalid after 3 attempts') from exc
                request = dict(request, format_retry=str(exc),
                    required_format={'skills': {p: 'non-empty full skill text' for p in paths}})

    @staticmethod
    def _repair_skills(reply, paths):
        if not isinstance(reply, dict):
            raise ValueError('CCV repair must be a JSON object')
        if 'skills' in reply:
            if set(reply) != {'skills'}:
                raise ValueError('Unexpected CCV repair wrapper fields')
            skills = reply['skills']
        else:
            skills = reply
        if not isinstance(skills, dict) or not skills:
            raise ValueError('CCV repair requires a non-empty skills mapping')
        if not set(skills) <= set(paths):
            raise ValueError('CCV repair scope violation')
        if any(not isinstance(text, str) or not text.strip() for text in skills.values()):
            raise ValueError('CCV repair text must be a non-empty string')
        return skills


class ScienceCCV(MathCCV):
    image = 'agent-overfit-continual-science:baseline-v1'
    attacks_module = 'continual_regularization.science_attacks'


class FinanceCCV(MathCCV):
    image = 'agent-overfit-continual-finance:baseline-v1'
    attacks_module = 'continual_regularization.finance_attacks'


class LawCCV(MathCCV):
    image = 'agent-overfit-continual-law:baseline-v1'
    attacks_module = 'continual_regularization.law_attacks'


class OfficeCCV(MathCCV):
    image = 'agent-overfit-continual-office:train-v1'
    attacks_module = 'continual_regularization.office_attacks'


class HealthcareCCV(MathCCV):
    image = 'agent-overfit-continual-healthcare:train-v1'
    attacks_module = 'continual_regularization.healthcare_attacks'
