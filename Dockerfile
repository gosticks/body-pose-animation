# FROM ubuntu:20.04
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
# RUN apt-get update && apt-get install -y wget build-essential

# install cuda and cudnn since it is somehow missing in nvidia/cuda image lol
# WORKDIR /tmp
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

# RUN wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.0-460.27.04-1_amd64.deb
# RUN dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.0-460.27.04-1_amd64.deb

# RUN apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub

#get deps
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git g++ make cmake wget libprotobuf-dev protobuf-compiler libopencv-dev \
    libgoogle-glog-dev libboost-all-dev libhdf5-dev libatlas-base-dev

# install python
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-dev python3-pip 

#for python api
# RUN pip3 install numpy opencv-python 

#get openpose
# WORKDIR /openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .

#build it
WORKDIR /openpose/build
RUN cmake -DBUILD_PYTHON=OFF .. && make -j `nproc`
WORKDIR /openpose
