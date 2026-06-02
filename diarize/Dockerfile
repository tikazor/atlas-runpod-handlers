FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /app

RUN pip install --no-cache-dir \
    "numpy<2" \
    runpod==1.6.2 \
    faster-whisper==1.0.3 \
    "pyannote.audio==3.1.1" \
    torchaudio==2.2.0

COPY handler.py .
CMD ["python", "-u", "handler.py"]