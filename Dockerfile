FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# packages
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends build-essential git nano rsync vim tree curl \
    wget unzip htop tmux xvfb patchelf ca-certificates bash-completion libjpeg-dev libpng-dev \
    ffmpeg cmake swig libssl-dev libcurl4-openssl-dev libopenmpi-dev python3-dev zlib1g-dev \
    qtbase5-dev qtdeclarative5-dev libglib2.0-0 libglu1-mesa-dev libgl1-mesa-dev libvulkan1 \
    libgl1-mesa-glx libosmesa6 libosmesa6-dev libglew-dev mesa-utils && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /root/.ssh

# miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda init && \
    conda clean -ya
ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# docker build . -t humanoid

# docker run -it --name humanoid-container -v /Users/hyeon/src/github/humanoid-bench-custom-task:/zero -t humanoid /bin/bash

# ## Installation
# Create a clean conda environment:
# ```
# conda create -n humanoidbench python=3.11
# conda activate humanoidbench
# ```
# Then, install the required packages:
# ```
# # Install HumanoidBench
# pip install -e .

# # jax GPU version
# pip install "jax[cuda12]==0.4.28"
# # Or, jax CPU version
# pip install "jax[cpu]==0.4.28"

# # Install jaxrl
# pip install -r requirements_jaxrl.txt

# # Install dreamer
# pip install -r requirements_dreamer.txt

# # Install td-mpc2
# pip install -r requirements_tdmpc.txt
# ```
WORKDIR /zero