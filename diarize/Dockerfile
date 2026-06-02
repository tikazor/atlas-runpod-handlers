FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

RUN pip install --no-cache-dir \
    "numpy<2" \
    "huggingface_hub==0.23.4" \
    "torch==2.2.0" \
    "torchaudio==2.2.0" \
    "faster-whisper==1.0.3" \
    "pyannote.audio==3.1.1" \
    "runpod==1.6.2"

COPY handler.py .
CMD ["python", "-u", "handler.py"]