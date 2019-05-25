#!/usr/bin/env bash
wrapper(){
    hour=$1
    command=$2
    #echo "Running: ${model_num}"
    module load python/3.6
    source $HOME/torchenv36/bin/activate
    module load scipy-stack
    echo $command > tmp.sh
    sed -i '1i\#!/bin/bash' tmp.sh
    sbatch  --job-name="${commend}" \
     --nodes=1  \
     --gres=gpu:1 \
     --cpus-per-task=6  \
     --mem=16000M \
     --time=0-${hour}:00 \
     --account=def-mpederso \
     --mail-user=jizong.peng.1@etsmtl.net \
     --mail-type=ALL   \
    ./tmp.sh
    rm ./tmp.sh
}
cd ..
time=1
# as in the paper
wrapper $time  "python script/train_IIC_Twohead.py Trainer.save_dir=runs/2head_5subhead"
# single head
wrapper $time  "python script/train_IIC_Twohead.py Trainer.save_dir=runs/1head_5subhead Trainer.head_control_params.A=0 Trainer.head_control_params.B=3"
# single head with 1 subhead
wrapper $time  "python script/train_IIC_Twohead.py Trainer.save_dir=runs/1head_1subhead Trainer.head_control_params.A=0 Trainer.head_control_params.B=3 Arch.num_sub_heads=1"
# two head with 1 subhead
wrapper $time  "python script/train_IIC_Twohead.py Trainer.save_dir=runs/2head_1subhead Arch.num_sub_heads=1"
# two head with two subhead of 30
wrapper $time  "python script/train_IIC_Twohead.py Trainer.save_dir=runs/2head_5subhead_k_30 Arch.num_sub_heads=5 Arch.output_k_A=30"