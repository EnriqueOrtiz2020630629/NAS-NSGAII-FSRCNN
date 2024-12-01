FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu122 \
    numpy \
    pillow \
    h5py \
    tqdm \
    matplotlib \
    pymoo 

CMD ["python3", "nas.py"]