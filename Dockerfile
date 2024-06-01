FROM pytorch/torchserve:0.11.0-gpu as builder

WORKDIR /app

COPY . ./

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv venv && \
    source .venv/bin/activate &&  \
    uv pip install -r requirements.txt