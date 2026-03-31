# Unified Dockerfile for nCompass examples
# Supports: PyTorch profiler, Nsight Systems (nsys), Nsight Compute (ncu)
#
# This single Dockerfile replaces all example-specific Dockerfiles and provides
# a consistent development environment with all profiling tools available.

ARG CUDA_VERSION=13.0.0

#################### BASE BUILD IMAGE ####################
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

# ARG CUDA_VERSION=12.9.1
ARG PYTHON_VERSION=3.10
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG VLLM_NIGHTLY=0

ENV DEBIAN_FRONTEND=noninteractive

# ============================================================
# STAGE 1: Base system packages
# ============================================================
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y \
        build-essential \
        ccache \
        software-properties-common \
        git curl sudo neovim wget \
        ninja-build \
        linux-tools-generic \
        libncurses-dev libncursesw5-dev \
    && ln -sf /usr/lib/linux-tools/*/perf /usr/bin/perf || true

# ============================================================
# STAGE 2: Python installation
# ============================================================
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-pip \
    && if [ "${PYTHON_VERSION}" != "3" ]; then \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1; \
    fi \
    && python3 --version

# ============================================================
# STAGE 3: NVIDIA Nsight Tools
# ============================================================
# Install Nsight Systems CLI
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends curl wget ca-certificates \
    && wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_2/NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb \
    && dpkg -i NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb \
    && rm NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb

# Install Nsight Compute
RUN apt-get update -y \
    && apt-get install -y cuda-nsight-compute-${CUDA_VERSION%.*}

# Install CUDA-GDB (full version)
RUN apt-get install -y cuda-gdb-${CUDA_VERSION%.*} \
    && ln -sf /usr/local/cuda/bin/cuda-gdb /usr/local/cuda/bin/cuda-gdb-full

# Verify installations
RUN (nsys --version || echo "Warning: nsys not found in PATH.") && \
    (ncu --version || echo "Warning: ncu not found in PATH.")

# ============================================================
# STAGE 4: User setup
# ============================================================
# Create user with host UID/GID for development (avoids permission issues)
# Handle case where UID/GID may already exist in base image
RUN existing_user=$(getent passwd ${HOST_UID} | cut -d: -f1 || echo ""); \
    if [ -n "$existing_user" ] && [ "$existing_user" != "ncuser" ]; then \
        usermod -l ncuser -d /home/ncuser -m "$existing_user"; \
        groupmod -n ncuser $(id -gn ncuser) 2>/dev/null || true; \
    elif [ -z "$existing_user" ]; then \
        groupadd -g ${HOST_GID} ncuser 2>/dev/null || true; \
        useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ncuser; \
    fi && \
    usermod -aG sudo ncuser && \
    usermod -s /bin/bash ncuser && \
    echo "ncuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ============================================================
# STAGE 5: Python virtual environment
# ============================================================
RUN mkdir -p /opt/venv && chown ${HOST_UID}:${HOST_GID} /opt/venv
RUN runuser -u ncuser -- python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CUDA library compatibility workarounds
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/
RUN ln -sf /usr/local/cuda/lib64/libcudart.so.$(echo $CUDA_VERSION | cut -d. -f1) /usr/local/cuda/lib64/libcudart.so

# Create cache directory and install uv package manager
RUN mkdir -p /home/ncuser/.cache && chown ${HOST_UID}:${HOST_GID} /home/ncuser/.cache
RUN --mount=type=cache,target=/home/ncuser/.cache/uv,uid=${HOST_UID},gid=${HOST_GID} \
    runuser -u ncuser -- pip install uv

# ============================================================
# STAGE 6: Rust toolchain (for trace_converter profiling)
# ============================================================
ENV RUSTUP_HOME=/home/ncuser/.rustup
ENV CARGO_HOME=/home/ncuser/.cargo
RUN runuser -u ncuser -- curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    runuser -u ncuser -- sh -s -- -y --default-toolchain stable \
    && chown -R ${HOST_UID}:${HOST_GID} /home/ncuser/.rustup /home/ncuser/.cargo
ENV PATH="/home/ncuser/.cargo/bin:${PATH}"
RUN runuser -u ncuser -- cargo install flamegraph

# ============================================================
# STAGE 7: Python dependencies
# ============================================================
COPY requirements/ /tmp/requirements/
RUN --mount=type=cache,target=/home/ncuser/.cache/uv,uid=${HOST_UID},gid=${HOST_GID} \
    if [ "$VLLM_NIGHTLY" = "1" ]; then BASE_REQS=base-nightly.txt; else BASE_REQS=base.txt; fi \
    && runuser -u ncuser -- uv pip install \
        -r /tmp/requirements/${BASE_REQS} \
        -r /tmp/requirements/dev.txt

# ============================================================
# STAGE 8: Environment and shell configuration
# ============================================================
ENV PYTHONPATH=/workspace
ENV PATH="/usr/local/nsight-systems/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

RUN echo "set -o vi" >> /home/ncuser/.bashrc \
    && echo "source /opt/venv/bin/activate" >> /home/ncuser/.bashrc \
    && echo 'export PATH="/home/ncuser/.cargo/bin:$PATH"' >> /home/ncuser/.bashrc \
    && echo 'alias cuda-gdb="/usr/local/cuda/bin/cuda-gdb-full"' >> /home/ncuser/.bashrc \
    && if [ "$VLLM_NIGHTLY" = "1" ]; then \
        echo "export LD_PRELOAD=/usr/local/cuda/lib64/libcublas.so.$(echo $CUDA_VERSION | cut -d. -f1):/usr/local/cuda/lib64/libcublasLt.so.$(echo $CUDA_VERSION | cut -d. -f1)" >> /home/ncuser/.bashrc; \
    fi \
    && chown ${HOST_UID}:${HOST_GID} /home/ncuser/.bashrc

# Switch to ncuser for default operation
USER ncuser
