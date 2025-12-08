# Dockerfile for nsys_example - converts nsys reports to Chrome trace format
# Based on NVIDIA CUDA base image with Nsight Systems installed

ARG CUDA_VERSION=13.0.0

#################### BASE BUILD IMAGE ####################
# Use NVIDIA CUDA base image with development tools
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

ARG CUDA_VERSION=13.0.0
ARG PYTHON_VERSION=3.10
ARG HOST_UID=1000
ARG HOST_GID=1000

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and basic utilities
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
    && apt-get install -y git curl sudo neovim wget

# Install pip
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

# Install NVIDIA Nsight Systems CLI
# Download and install the latest Nsight Systems CLI package
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends curl wget ca-certificates \
    && wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_2/NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb \
    && dpkg -i NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb

# Verify nsys installation
RUN (nsys --version || echo "Warning: nsys not found in PATH. Ensure it's installed correctly.")

# Install packages as ncuser to maintain correct ownership
# Create .cache directory with correct ownership for uv
RUN mkdir -p /home/ncuser/.cache && chown ${HOST_UID}:${HOST_GID} /home/ncuser/.cache
RUN --mount=type=cache,target=/home/ncuser/.cache/uv,uid=${HOST_UID},gid=${HOST_GID} \
  runuser -u ncuser -- pip install uv

RUN ln -sf /usr/local/cuda/lib64/libcudart.so.13 /usr/local/cuda/lib64/libcudart.so

# Set up environment
ENV PYTHONPATH=/workspace

# Add nsys to PATH if installed in non-standard location
ENV PATH="/usr/local/nsight-systems/bin:${PATH}"

RUN echo "set -o vi" >> ~/.bashrc
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc

# Copy bashrc configuration to ncuser
RUN cp /root/.bashrc /home/ncuser/.bashrc \
    && chown ${HOST_UID}:${HOST_GID} /home/ncuser/.bashrc

# Switch to ncuser for default operation
USER ncuser
