FROM nvcr.io/nvidia/pytorch:24.04-py3
#FROM nvidia/cuda:12.1.1-devel-ubuntu20.04 AS app
#FROM nvcr.io/nvidia/pytorch:24.11-py3
#FROM nvcr.io/nvidia/pytorch:22.10-py3
#FROM nvcr.io/nvidia/pytorch:21.09-py3

ARG conda_ver="py311_24.5.0-0-Linux-x86_64"

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
      wget curl

#RUN git clone --recurse-submodules https://github.com/onnx/onnx-tensorrt.git \
#    && cd onnx-tensorrt && mkdir build && cd build \
#    && cmake .. && make install \
#    && cd ../.. && rm -rf onnx-tensorrt

# Install Miniconda
ENV CONDA_HOME=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-${conda_ver}.sh \
    && bash Miniconda3-${conda_ver}.sh -b -p ${CONDA_HOME}
ENV PATH=/home/user/.local/bin:${CONDA_HOME}/bin:$PATH

# Install dependencies
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install torch torchvision torchaudio
RUN pip install timm altair duckdb gcsfs tensorflow tf2onnx scikit-learn efficientnet_pytorch

# Install DBTNet
COPY . /dbtnet
RUN cd /dbtnet && \
    pip install --no-cache-dir -e .

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH=$PATH:/usr/local/gcloud/google-cloud-sdk/bin
