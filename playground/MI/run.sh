#!/usr/bin/env bash
set -e
ARCHIVE_PATH='archives/'
current_path=$(pwd)
project_path=$(cd ../.. && pwd)
cd $project_path
source deepclustering/utils/bash_utils.sh
cd $current_path

python main.py Trainer.save_dir=MI_MNIST/IMSAT/use_vat Trainer.name=IMSAT Trainer.use_vat=True &
python main.py Trainer.save_dir=MI_MNIST/IMSAT/use_rt Trainer.name=IMSAT Trainer.use_vat=False &

python main.py Trainer.save_dir=MI_MNIST/IIC/use_vat Trainer.name=IIC Trainer.sat_weight=0.1 &
python main.py Trainer.save_dir=MI_MNIST/IIC/baseline Trainer.name=IIC Trainer.sat_weight=0.0 &

wait_script
cd ../..
python deepclustering/postprocessing/plot.py --folders $ARCHIVE_PATH/MI_MNIST/IMSAT/use_vat \
$ARCHIVE_PATH/MI_MNIST/IMSAT/use_rt \
$ARCHIVE_PATH/MI_MNIST/IIC/use_vat \
$ARCHIVE_PATH/MI_MNIST/IIC/baseline \
--file=wholeMeter.csv --out_dir=$ARCHIVE_PATH/MI_MNIST
zip -r $ARCHIVE_PATH/MI_MNIST.zip $ARCHIVE_PATH/MI_MNIST/
