# Dockerfile for basic_example
# Provides environment for PyTorch profiling with ncompass

ARG CUDA_VERSION=12.9.1

#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

ARG CUDA_VERSION=12.9.1
ARG PYTHON_VERSION=3.10
ARG HOST_UID=1000
ARG HOST_GID=1000

ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && if [ "${PYTHON_VERSION}" != "3" ]; then update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1; fi \
    && python3 --version

RUN apt-get update -y \
    && apt-get install -y git curl sudo neovim

# Install pip s.t. it will be compatible with our PYTHON_VERSION
RUN apt-get install -y python3-pip

# Create user with host UID/GID for development (avoids permission issues)
# Handle case where UID/GID may already exist in base image (e.g., ubuntu user with UID 1000)
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

# Create /opt/venv directory with correct ownership BEFORE creating venv
# This way the venv will be created with correct ownership from the start
RUN mkdir -p /opt/venv && chown ${HOST_UID}:${HOST_GID} /opt/venv

# Create venv as ncuser so it's owned by the correct user from the start
RUN runuser -u ncuser -- python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# Install packages as ncuser to maintain correct ownership
RUN --mount=type=cache,target=/home/ncuser/.cache/pip \
  runuser -u ncuser -- pip install uv

RUN ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so

RUN echo "set -o vi" >> ~/.bashrc
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc

COPY requirements.txt /tmp/requirements.txt
RUN chown ncuser:ncuser /tmp/requirements.txt && \
    runuser -u ncuser -- uv pip install -r /tmp/requirements.txt

# Copy bashrc configuration to ncuser
RUN cp /root/.bashrc /home/ncuser/.bashrc \
    && chown ${HOST_UID}:${HOST_GID} /home/ncuser/.bashrc

# Switch to ncuser for default operation
USER ncuser

ENV PYTHONPATH=/workspace
