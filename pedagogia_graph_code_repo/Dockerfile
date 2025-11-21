FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        clang \
        llvm-dev \
        build-essential \
        curl \
        git \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install requirements and libclang wheel instead of the mismatched 'clang' binding
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir libclang

COPY . /app
CMD ["python"]
