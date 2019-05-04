#!/usr/bin/env bash
cd ..
docker build -t containerimage . && docker run --name testcontainer --runtime=nvidia --rm containerimage nvidia-smi && pytest