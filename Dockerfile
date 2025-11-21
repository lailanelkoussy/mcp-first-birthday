FROM python:3.13-slim

# --- Environment variables for better logging ---
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# --- System dependencies ---
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        clang \
        llvm-dev \
        build-essential \
        curl \
        git \
        pkg-config \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Install Python dependencies ---
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir libclang


# --- Copy your application ---
COPY . /app

# --- Expose ports ---
# HF Spaces wants 7860 for Gradio, but Weaviate uses 8080 + 50051
EXPOSE 7860 8080 50051

# --- Entrypoint ---
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
