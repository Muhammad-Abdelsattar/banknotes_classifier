FROM python:3.10-slim
COPY app/ /app/
COPY artifacts/ /app/artifacts/
WORKDIR /app

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

ENTRYPOINT [ "python", "main.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]