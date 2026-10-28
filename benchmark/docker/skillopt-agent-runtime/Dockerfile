FROM agent-runtime:latest

# Freeze the spreadsheet runtime before any sample starts.  Agents therefore
# never need to mutate shared /tmp or install packages during an experiment.
RUN python3 -m venv /opt/skillopt-python \
    && /opt/skillopt-python/bin/pip install --no-cache-dir \
        openpyxl==3.1.5 pandas==2.2.3 \
    && ln -sf /opt/skillopt-python/bin/python /usr/local/bin/python \
    && ln -sf /opt/skillopt-python/bin/python3 /usr/local/bin/python3 \
    && ln -sf /opt/skillopt-python/bin/pip /usr/local/bin/pip

ENV PATH="/opt/skillopt-python/bin:/root/.local/bin:${PATH}"
