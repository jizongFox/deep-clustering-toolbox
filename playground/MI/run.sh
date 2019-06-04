#!/usr/bin/env bash
set -e
ARCHIVE_PATH='archives/'
current_path=$(pwd)
project_path=$(cd ../.. && pwd)
cd ${project_path}
source deepclustering/utils/bash_utils.sh
cd ${current_path}
set -e

weight=0.1
python main.py Trainer.save_dir=MI_MNIST/IMSAT/use_vat_${weight} Trainer.name=IMSAT Trainer.use_vat=True Trainer.sat_weight=${weight} &# Trainer.checkpoint_path=${project_path}/runs/MI_MNIST/IMSAT/use_vat &
python main.py Trainer.save_dir=MI_MNIST/IMSAT/use_rt_${weight} Trainer.name=IMSAT Trainer.use_vat=False Trainer.sat_weight=${weight} &#Trainer.checkpoint_path=${project_path}/runs/MI_MNIST/IMSAT/use_rt &

wait_script

python main.py Trainer.save_dir=MI_MNIST/IIC/baseline Trainer.name=IIC Trainer.sat_weight=0.0 & #Trainer.checkpoint_path=${project_path}/runs/MI_MNIST/IIC/baseline &
python main.py Trainer.save_dir=MI_MNIST/IIC/use_vat_${weight} Trainer.name=IIC Trainer.sat_weight=$weight & #Trainer.checkpoint_path=${project_path}/runs/MI_MNIST/IIC/use_vat &

wait_script
cd ../..
python deepclustering/postprocessing/plot.py --folders ${ARCHIVE_PATH}/MI_MNIST/IMSAT/use_vat_${weight} \
${ARCHIVE_PATH}/MI_MNIST/IMSAT/use_rt_${weight} \
${ARCHIVE_PATH}/MI_MNIST/IIC/use_vat_${weight} \
${ARCHIVE_PATH}/MI_MNIST/IIC/baseline \
--file=wholeMeter.csv --out_dir=${ARCHIVE_PATH}/MI_MNIST
zip -rq ${ARCHIVE_PATH}/MI_MNIST.zip ${ARCHIVE_PATH}/MI_MNIST/
