FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime
RUN apt-get update && apt-get install -y --no-install-recommends
#        build-essential \
#        nano \
#        python3-setuptools \
#        python3-pip \
#    && rm -rf /var/lib/apt/lists/* \
#    && pip3 install --upgrade pip
COPY . /workspace
WORKDIR /workspace
RUN ls && pip install -r requirment.txt &&  pip install -e .
#docker build -t containerimage . && docker run --name testcontainer --runtime=nvidia --rm containerimage nvidia-smi && pytest