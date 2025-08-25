FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y wget \
    libxml2 \
    cuda-minimal-build-12-2 \
    libcusparse-dev-12-2 \
    libcublas-dev-12-2 \
    libcusolver-dev-12-2 \
    cuda-toolkit-12.2 \
    git \
    libgfortran5 

RUN wget -P /tmp \
    "https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Miniforge3-Linux-x86_64.sh" \ 
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh
ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_OVERRIDE_CUDA=12.2
RUN conda install pip pandas matplotlib numpy"<2.0.0" biopython scipy pdbfixer seaborn tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku dm-tree joblib ml-collections immutabledict optax jaxlib=*=*cuda* jax cuda-nvcc cudnn -c conda-forge -c anaconda -c nvidia  --channel https://conda.graylab.jhu.edu -y -n base

# install ColabDesign
RUN pip3 install git+https://github.com/sokrypton/ColabDesign.git --no-deps

# Download AlphaFold2 weights
RUN mkdir -p /params/ && cd /params/ && wget -P . https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar && tar -xvf alphafold_params_2022-12-06.tar && cd ..

COPY . /opt/BindCraft
RUN chmod +x /opt/BindCraft/functions/dssp
RUN chmod +x /opt/BindCraft/functions/DAlphaBall.gcc

ENTRYPOINT ["python", "-u", "/opt/BindCraft/bindcraft.py" ]