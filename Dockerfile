FROM pytorch/torchserve:0.11.0-gpu as builder

WORKDIR /app

COPY . ./

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv venv && \
    source .venv/bin/activate &&  \
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN uv pip install nvidia-cublas-cu11 nvidia-cudnn-cu11 && \
    uv pip install -r requirements.txt