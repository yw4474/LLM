FROM python:3.12-slim-bookworm

WORKDIR /code

# system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /code/requirements.txt

# 1) Install CPU-only torch FIRST from PyTorch index
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# 2) Install the rest from default PyPI
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY . /code

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
