# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/source/dev/dockerfile/dockerfile.rst and
# docs/source/assets/dev/dockerfile-stages-dependency.png

ARG CUDA_VERSION=12.9.1

#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

ARG CUDA_VERSION=12.9.1
ARG PYTHON_VERSION=3.12

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
RUN python${PYTHON_VERSION} -m venv /opt/venv 
ENV PATH="/opt/venv/bin:$PATH"

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /workspace

RUN --mount=type=cache,target=/root/.cache/pip \
  pip install uv

RUN ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so

RUN echo "set -o vi" >> ~/.bashrc
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc

COPY requirements.txt /workspace/requirements.txt
RUN uv pip install -r /workspace/requirements.txt 

# Accept vLLM wheel filename via build-arg and copy wheels from build_whls/
ENV PYTHONPATH=/workspace
