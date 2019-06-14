#!/usr/bin/env bash
wrapper(){
    hour=$1
    command=$2
    module load python/3.6
    source $HOME/torchenv36/bin/activate
    module load scipy-stack
	module load cmake
    echo ${command} > tmp.sh
    sed -i '1i\#!/bin/bash' tmp.sh
    sbatch  --job-name="${commend}" \
     --nodes=1  \
     --gres=gpu:1 \
     --cpus-per-task=6  \
     --mem=16000M \
     --time=0-${hour}:00 \
     --account=rrg-mpederso \
     --mail-user=jizong.peng.1@etsmtl.net \
     --mail-type=ALL   \
    ./tmp.sh
    rm ./tmp.sh
}
