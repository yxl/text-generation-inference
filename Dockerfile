# Rust builder
FROM lukemathwalker/cargo-chef:latest-rust-1.71 AS chef
WORKDIR /usr/src

ARG CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

# cargo 国内镜像加速
RUN mkdir -p .cargo
COPY config.toml .cargo/
ENV RUSTUP_DIST_SERVER="https://mirrors.ustc.edu.cn/rust-static"
ENV RUSTUP_UPDATE_ROOT="https://mirrors.ustc.edu.cn/rust-static/rustup"

FROM chef as planner
COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY benchmark benchmark
COPY router router
COPY launcher launcher
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

ARG GIT_SHA
ARG DOCKER_LABEL

RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl --progress-bar -OL https://ghproxy.com/https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY --from=planner /usr/src/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY benchmark benchmark
COPY router router
COPY launcher launcher
RUN cargo build --release

# Python builder
# Adapted from: https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM nvidia/cuda:11.8.0-base-ubuntu20.04 as pytorch-install

ARG PYTORCH_VERSION=2.0.1
ARG PYTHON_VERSION=3.9
# Keep in sync with `server/pyproject.toml
ARG CUDA_VERSION=11.8
# Automatically set by buildx
ARG TARGETPLATFORM

ENV PATH /opt/conda/bin:$PATH

# 检测显卡
RUN nvidia-smi -L || { echo "未检测到显卡,如果显卡正确配置,使用DOCKER_BUILDKIT=0重新build"; exit 1; }

# 使用国内源加速
RUN sed -i "s@\(archive\|security\).ubuntu.com@mirrors.aliyun.com@g" /etc/apt/sources.list
RUN echo >>/etc/apt/apt.conf.d/99verify-peer.conf "Acquire { https::Verify-Peer false }"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        git && \
        rm -rf /var/lib/apt/lists/*

# conda 国内镜像加速
COPY .condarc /root/.condarc

# Install conda
# translating Docker's TARGETPLATFORM into mamba arches
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  MAMBA_ARCH=aarch64  ;; \
         *)              MAMBA_ARCH=x86_64   ;; \
    esac && \
    curl --progress-bar -fsSL -v -o ~/mambaforge.sh -O  "https://ghproxy.com/https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-${MAMBA_ARCH}.sh"
RUN chmod +x ~/mambaforge.sh && \
    bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh

RUN /opt/conda/bin/conda install -n base conda-libmamba-solver && /opt/conda/bin/conda clean -ya
RUN /opt/conda/bin/conda config --set solver libmamba && /opt/conda/bin/conda clean -ya
RUN /opt/conda/bin/conda install -y "python=${PYTHON_VERSION}" -c conda-forge && /opt/conda/bin/conda clean -ya

# Install pytorch adn cuda
RUN /opt/conda/bin/conda install -c "nvidia/label/cuda-11.8.0" -c torch -c nvidia -y pytorch==$PYTORCH_VERSION pytorch-cuda=$CUDA_VERSION cuda==$CUDA_VERSION && /opt/conda/bin/conda clean -ya

ENV CUDA_HOME='/opt/conda'

# CUDA kernels builder image
FROM pytorch-install as kernel-builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ninja-build \
        && rm -rf /var/lib/apt/lists/*

# pip 国内加速
RUN pip config set global.index-url https://mirrors.tencentyun.com/pypi/simple

# fix fatal error: pybind11/pybind11.h: No such file or directory
RUN pip install "pybind11[global]" --no-cache-dir

# 检查 cuda 是否可用
RUN python -c "import torch;print('CUDA is READY') if torch.cuda.is_available() else exit(0)"

# Build Flash Attention CUDA kernels
FROM kernel-builder as flash-att-builder

WORKDIR /usr/src

COPY server/Makefile-flash-att Makefile

RUN ls /opt/conda

# Build specific version of flash attention
RUN make build-flash-attention

# Build Flash Attention v2 CUDA kernels
FROM kernel-builder as flash-att-v2-builder

WORKDIR /usr/src

COPY server/Makefile-flash-att-v2 Makefile

# Build specific version of flash attention v2
RUN make build-flash-attention-v2

# Build Transformers exllama kernels
FROM kernel-builder as exllama-kernels-builder

WORKDIR /usr/src

COPY server/exllama_kernels/ .

# Build specific version of transformers
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" python setup.py build

# Build Transformers CUDA kernels
FROM kernel-builder as custom-kernels-builder

WORKDIR /usr/src

COPY server/custom_kernels/ .

# Build specific version of transformers
RUN python setup.py build

# Build vllm CUDA kernels
FROM kernel-builder as vllm-builder

WORKDIR /usr/src

COPY server/Makefile-vllm Makefile

# Build specific version of vllm
RUN make build-vllm

# Text Generation Inference base image
FROM nvidia/cuda:11.8.0-base-ubuntu20.04 as base

# Conda env
ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

# Text Generation Inference base env
ENV HUGGINGFACE_HUB_CACHE=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80

WORKDIR /usr/src

# 设置为中国国内源
RUN sed -i "s@\(archive\|security\).ubuntu.com@mirrors.aliyun.com@g" /etc/apt/sources.list
RUN echo >>/etc/apt/apt.conf.d/99verify-peer.conf "Acquire { https::Verify-Peer false }"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libssl-dev \
        ca-certificates \
        make \
        && rm -rf /var/lib/apt/lists/*

# Copy conda with PyTorch installed
COPY --from=pytorch-install /opt/conda /opt/conda

# Copy build artifacts from flash attention builder
COPY --from=flash-att-builder /usr/src/flash-attention/build/lib.linux-x86_64-cpython-39 /opt/conda/lib/python3.9/site-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/layer_norm/build/lib.linux-x86_64-cpython-39 /opt/conda/lib/python3.9/site-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/rotary/build/lib.linux-x86_64-cpython-39 /opt/conda/lib/python3.9/site-packages

# Copy build artifacts from flash attention v2 builder
COPY --from=flash-att-v2-builder /usr/src/flash-attention-v2/build/lib.linux-x86_64-cpython-39 /opt/conda/lib/python3.9/site-packages

# Copy build artifacts from custom kernels builder
COPY --from=custom-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-39 /opt/conda/lib/python3.9/site-packages
# Copy build artifacts from exllama kernels builder
COPY --from=exllama-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-39 /opt/conda/lib/python3.9/site-packages

# Copy builds artifacts from vllm builder
COPY --from=vllm-builder /usr/src/vllm/build/lib.linux-x86_64-cpython-39 /opt/conda/lib/python3.9/site-packages

# pip 国内加速
RUN pip config set global.index-url https://mirrors.tencentyun.com/pypi/simple

# Install flash-attention dependencies
RUN pip install einops --no-cache-dir


# Install benchmarker
COPY --from=builder /usr/src/target/release/text-generation-benchmark /usr/local/bin/text-generation-benchmark
# Install router
COPY --from=builder /usr/src/target/release/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=builder /usr/src/target/release/text-generation-launcher /usr/local/bin/text-generation-launcher

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        && rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        && rm -rf /var/lib/apt/lists/*

# for qwen int4 model
RUN pip install auto-gptq tiktoken transformers_stream_generator --no-cache-dir

# Install server
COPY proto proto
COPY server server
COPY server/Makefile server/Makefile
RUN cd server && \
    make gen-server && \
    pip install -r requirements.txt --no-cache-dir && \
    pip install ".[bnb, accelerate, quantize]" --no-cache-dir

# AWS Sagemaker compatbile image
FROM base as sagemaker

COPY sagemaker-entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# Final image
FROM base

ENTRYPOINT ["text-generation-launcher"]
CMD ["--json-output"]
