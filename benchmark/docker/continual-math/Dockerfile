FROM agent-overfit-skillopt-runtime:latest

# Only the new image's writable layers change; the parent image is untouched.
RUN python3 -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 statsmodels==0.14.4 \
    sympy==1.13.3 openpyxl==3.1.5 requests==2.32.3 pytest==8.4.1 \
    openai==2.45.0 litellm==1.80.8

WORKDIR /root
RUN mkdir -p /app /logs/verifier /root/data /verifier /app/workspace \
    && test ! -f /root/.codex/auth.json \
    && test ! -d /root/.codex/sessions \
    && test ! -e /root/tasks_spec.json

COPY skills /root/.agents/skills
COPY skills /root/.claude/skills
COPY skills /root/.terminus/skills
COPY skills /root/.codex/skills
COPY skills /root/.opencode/skill
COPY skills /root/.goose/skills
COPY skills /root/.factory/skills
COPY skills /root/.gemini/skills

RUN codex --version && python3 -m pip check
