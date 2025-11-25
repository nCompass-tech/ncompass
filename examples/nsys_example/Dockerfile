# Dockerfile for nsys_example - converts nsys reports to Chrome trace format
# Based on NVIDIA CUDA base image with Nsight Systems installed

ARG CUDA_VERSION=12.9.1

#################### BASE BUILD IMAGE ####################
# Use NVIDIA CUDA base image with development tools
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

ARG CUDA_VERSION=12.9.1
ARG PYTHON_VERSION=3.12
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
RUN python${PYTHON_VERSION} -m venv /opt/venv 
ENV PATH="/opt/venv/bin:$PATH"

# Install NVIDIA Nsight Systems CLI
# Download and install the latest Nsight Systems CLI package
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends curl wget ca-certificates \
    && wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_2/NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb \
    && dpkg -i NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb \
    || (echo "Warning: Failed to install nsys from direct download. Trying repository method..." \
    && apt-get install -y gnupg \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
    && echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64 /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-get update -y \
    && apt-get install -y nsight-systems-cli || echo "Note: nsys installation failed. You may need to install manually.")

# Verify nsys installation
RUN (nsys --version || echo "Warning: nsys not found in PATH. Ensure it's installed correctly.")

# Install uv for faster pip operations
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install uv

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install -r /tmp/requirements.txt 

# Set up environment
ENV PYTHONPATH=/workspace

# Add nsys to PATH if installed in non-standard location
ENV PATH="/usr/local/nsight-systems/bin:${PATH}"

RUN echo "set -o vi" >> ~/.bashrc
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc

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

# Set permissions for venv
RUN chown -R ${HOST_UID}:${HOST_GID} /opt/venv

# Copy bashrc configuration to ncuser
RUN cp /root/.bashrc /home/ncuser/.bashrc \
    && chown ${HOST_UID}:${HOST_GID} /home/ncuser/.bashrc

# Switch to ncuser for default operation
USER ncuser
