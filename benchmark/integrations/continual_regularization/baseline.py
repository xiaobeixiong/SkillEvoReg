"""Official CodexSeq with transport safety only; no Regularization or learning changes."""
import json
import os
import shlex
import re
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from harbor.agents.installed.codex_seq import CodexSeq

DIRECT_PREFIX = "unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; export NO_PROXY='*' no_proxy='*'; "
API_PREFLIGHT = """import os
import httpx
from openai import OpenAI
with httpx.Client(trust_env=False, timeout=45) as transport:
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ['OPENAI_BASE_URL'], http_client=transport, max_retries=0)
    response = client.responses.create(model='gpt-5.4-mini', input='Reply OK.', reasoning={'effort':'medium'}, max_output_tokens=128)
    assert response.status in ('completed', 'incomplete'), 'API preflight failed'
    print('PASS: direct authenticated Responses API; model='+response.model, flush=True)
"""

def codex_route_flags():
    """Pin the CLI provider explicitly, including on resume commands."""
    values = {
        "model_provider": "continual_yibu",
        "model_providers.continual_yibu.name": "Yibu",
        "model_providers.continual_yibu.base_url": os.environ["OPENAI_BASE_URL"],
        "model_providers.continual_yibu.env_key": "OPENAI_API_KEY",
        "model_providers.continual_yibu.wire_api": "responses",
    }
    return " ".join("-c " + shlex.quote(k + "=" + json.dumps(v)) for k, v in values.items())

class CheckedEnvironment:
    def __init__(self, environment, logs_dir, display=None, plan=None):
        self.environment = environment
        self.logs_dir = logs_dir
        self.index = 0
        self.display = display
        self.plan = plan if plan is not None else {}
        self.scores = {}

    def __getattr__(self, name):
        return getattr(self.environment, name)

    async def exec(self, **kwargs):
        self.index += 1
        number, phase = self.plan.get(self.index, (0, "初始化"))
        if self.display:
            progress, task = self.display
            progress.update(task, phase=f"第 {number} 题 · {phase}" if number else phase)
        result = await self.environment.exec(**kwargs)
        if result.return_code != 0:
            (self.logs_dir / "infrastructure_failure.json").write_text(json.dumps({
                "command_index": self.index, "return_code": result.return_code,
                "stdout": result.stdout, "stderr": result.stderr,
            }, ensure_ascii=False))
            raise RuntimeError("Continual command failed; stopping before further skill updates")
        if "HARBOR_GRADING_SPEC_JSON" in (kwargs.get("env") or {}):
            for line in (result.stdout or "").splitlines():
                if line.startswith("CONTINUAL_SCORE "):
                    data = json.loads(line[len("CONTINUAL_SCORE "):])
                    self.scores[data["task_id"]] = float(data["score"])
            if self.display and self.scores:
                mean = sum(self.scores.values()) / len(self.scores)
                progress.update(task, score=f"已评分 {len(self.scores)} 题均分 {mean:.1%}")
        if self.display and phase == "确认技能写入":
            progress.update(task, completed=number, phase=f"第 {number} 题完成")
        return result


class MathBaseline(CodexSeq):
    display_name = "Math"

    def __init__(self, *args, resume_checkpoint=None, allow_legacy_recovery=False,
                 enable_checkpoints=True, **kwargs):
        self.resume_checkpoint = resume_checkpoint
        self.allow_legacy_recovery = allow_legacy_recovery
        self.enable_checkpoints = enable_checkpoints
        super().__init__(*args, **kwargs)

    def _build_judge_run_command(self, task_num, judge_task_id):
        # Never interpolate credentials into command.txt. They are exec env only.
        path = f"/logs/verifier/task_{judge_task_id:03d}_feedback.json"
        check = (
            "import json; from pathlib import Path; "
            f"d=json.loads(Path({path!r}).read_text()); "
            "assert not d.get('judge_error'), 'Judge API failure: batch not scoreable'; "
            f"print('CONTINUAL_SCORE '+json.dumps(dict(task_id={judge_task_id}, "
            "score=d.get('score',d.get('total_score',0)))))"
        )
        return (
            "mkdir -p /logs/verifier && python3 /root/judge_subtask.py "
            f"--task-id {judge_task_id} --output {path} "
            f"--model {shlex.quote(self._judge_model)} "
            f"--pass-threshold {self._judge_pass_threshold} && "
            f"python3 -c {shlex.quote(check)} && "
            "rm -rf /tests"
        )

    def create_run_agent_commands(self, instruction):
        commands = super().create_run_agent_commands(instruction)
        number = 0
        self._checkpoint_boundaries = set()
        plan = getattr(self, "_display_plan", {})
        for command_index, item in enumerate(commands, 1):
            # The official image preinstalls Codex but has no nvm.
            command = item.command.replace(". ~/.nvm/nvm.sh; ", "")
            if command.startswith("codex exec"):
                prefix = "codex exec resume --last" if command.startswith("codex exec resume --last") else "codex exec"
                command = command.replace(prefix, prefix + " " + codex_route_flags(), 1)
            item.command = "bash -o pipefail -c " + shlex.quote(DIRECT_PREFIX + command)
            item.env = dict(item.env or {})
            item.env["OPENAI_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
            item.env["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]
            item.env.update(NO_PROXY="*", no_proxy="*")
            if command.startswith("codex exec"):
                match = re.search(r"\[Task (\d+) of \d+\]", command)
                if match:
                    number = int(match.group(1))
                    phase = "做题"
                else:
                    phase = "技能更新" if number else "初始化会话"
            elif item.sync_task_tests_dir:
                phase = "评分"
            elif command.startswith("echo '=== skill verification"):
                phase = "确认技能写入"
            else:
                phase = "保存上下文" if number else "准备环境"
            plan[command_index] = (number, phase)
            if command.startswith('python3 /root/append_task_memory.py '):
                self._checkpoint_boundaries.add(command_index)
        # Last task has no following memory command. Its verification boundary
        # comes after Regularization commit, and before credential cleanup.
        verification = [i for i, (_, phase) in plan.items() if phase == '确认技能写入']
        if verification:
            self._checkpoint_boundaries.add(verification[-1])
        self._checkpoint_total_steps = number
        self._display_plan = plan
        return commands

    async def run(self, instruction, environment, context):
        from harbor.agents.installed.codex_seq import _split_tasks
        print(f'[{self.display_name} preflight] checking direct authenticated API (45s timeout)', flush=True)
        probe = await environment.exec(
            command='python3 -c '+shlex.quote(API_PREFLIGHT),
            env={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
                 "OPENAI_BASE_URL": os.environ["OPENAI_BASE_URL"],
                 "NO_PROXY": "*", "no_proxy": "*"}, timeout_sec=60)
        if probe.return_code:
            (self.logs_dir / 'preflight_error.txt').write_text((probe.stderr or '')+(probe.stdout or ''))
            raise RuntimeError('Container API preflight failed; no training started')
        print((probe.stdout or '').strip(), flush=True)
        self._display_plan = {}
        if self.enable_checkpoints or self.resume_checkpoint:
            from .checkpoint import CheckpointEnvironment
            environment = CheckpointEnvironment(environment, self, instruction)
        with Progress(TextColumn(self.display_name), BarColumn(),
                      TextColumn("{task.completed:.0f}/{task.total:.0f}"),
                      TextColumn("{task.fields[phase]}"),
                      TextColumn("{task.fields[score]}"), TimeElapsedColumn()) as progress:
            task = progress.add_task(self.display_name, total=len(_split_tasks(instruction)),
                                     phase="初始化", score="尚无评分")
            await super().run(instruction, CheckedEnvironment(
                environment, self.logs_dir, (progress, task), self._display_plan), context)


class ScienceBaseline(MathBaseline):
    """The same transport-safe official updater on the Science sequence."""

    display_name = "Science"


class FinanceBaseline(MathBaseline):
    """The transport-safe official updater on the Finance/Economics sequence."""

    display_name = "Finance"


class LawBaseline(MathBaseline):
    """The transport-safe official updater on the published Law sequence."""

    display_name = "Law"


class OfficeBaseline(MathBaseline):
    """The transport-safe official updater on the published Office sequence."""

    display_name = "Office"


class HealthcareBaseline(MathBaseline):
    """The transport-safe official updater on the published Healthcare sequence."""

    display_name = "Healthcare"
