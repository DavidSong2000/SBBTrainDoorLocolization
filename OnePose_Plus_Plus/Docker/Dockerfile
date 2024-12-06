FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -qq && \
    apt upgrade -y -qq && \
    apt install -y -qq \
    	python3 \
	python-is-python3 \
	python3-pip \
	git \
	build-essential \
	cmake \
	ninja-build \
	wget \
	nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install colmap Dependencies
RUN apt update -qq && \
    apt install -y -qq \
	libboost-program-options-dev \
	libboost-filesystem-dev \
	libboost-graph-dev \
	libeigen3-dev \	
	libflann-dev \
	libsqlite3-dev \
	libceres-dev \
	libcgal-dev \
	libfreeimage-dev \
	libmetis-dev \
	libglew-dev \
    && rm -rf /var/lib/apt/lists/*


ARG COLMAP_VERSION=3.9.1

# Build and install COLMAP.
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    git reset --hard ${COLMAP_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DGUI_ENABLED=OFF -DCMAKE_CUDA_ARCHITECTURES=61 && \
    ninja && \
    ninja install && \
    cd .. && rm -rf colmap

ADD requirements.txt /

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@stable"
