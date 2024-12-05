# 使用基础 Ubuntu 镜像
FROM ubuntu:20.04 as intermediate

# install git
RUN apt-get update && \
    apt-get install -y git build-essential python3-dev curl libatlas-base-dev libhdf5-dev python3-pip

# 使用 curl 下载 get-pip.py
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

# add credentials on build
ARG SSH_PRIVATE_KEY
RUN mkdir -p /root/.ssh/ && \
    echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_ed25519 && \
    chmod 600 /root/.ssh/id_ed25519
RUN cat /root/.ssh/id_ed25519


# make sure your domain is accepted
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN git clone --recursive git@github.com:cvg/limap.git

# From here, final image
FROM ubuntu:20.04

ARG COLMAP_VERSION=3.8

# Prevent stop building ubuntu at time zone selection.  
ENV DEBIAN_FRONTEND=noninteractive

# install git
RUN apt-get update && \
    apt-get install -y git build-essential python3-dev curl libatlas-base-dev libhdf5-dev python3-pip

# 使用 curl 下载 get-pip.py
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

# Prepare and empty machine for building.
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    wget \
    software-properties-common \
    lsb-core

# 使用 Kitware 的仓库安装最新的 CMake 版本
RUN apt-get update && \
    apt-get install -y wget gnupg && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
    apt-get update && \
    apt-get install -y cmake

# 安装 pip 和 setuptools
RUN curl -O https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py


# 安装适合 M1 的 PyTorch
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/arm64/torch_stable.html


# Build and install COLMAP.
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    git reset --hard ${COLMAP_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja && \
    ninja && \
    ninja install && \
    cd .. && rm -rf colmap

# Enable GUI
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

# PoseLib Dependency for LIMAP
RUN git clone --recursive https://github.com/vlarsson/PoseLib.git && \
    cd PoseLib && \
    mkdir build && cd build && \
    cmake .. && \
    make install -j8

RUN apt-get update && \
    apt-get install -y \
    libhdf5-dev

# Only Python 3.9 seems to satisfy all dependencies
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y --fix-missing\
    python3.9-dev \
    python3.9-venv

ENV VIRTUAL_ENV=/opt/venv
RUN /usr/bin/python3.9 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"



COPY --from=intermediate /limap /limap
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --upgrade pip setuptools && \
    cd limap && \
    python --version && \
    pip --version && \
    python -m pip install pyyaml tqdm attrdict h5py numpy scipy matplotlib seaborn brewer2mpl \
        tensorboard tensorboardX opencv-python opencv-contrib-python scikit-learn scikit-image \
        shapely jupyter bresenham omegaconf rtree plyfile pathlib open3d==0.16.0 imagesize \
        einops ninja yacs python-json-logger ruff==0.6.7 && \
    python -m pip install -Ive .
