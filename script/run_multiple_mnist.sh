#!/usr/bin/env bash
CURRENT_PATH=$(pwd)
PROJECT_PATH=$(python -c "from deepclustering import PROJECT_PATH; print(PROJECT_PATH)")
WRAPPER_PATH=${PROJECT_PATH}"/deepclustering/utils/CC_wrapper.sh"
echo "The project path: ${PROJECT_PATH}"
echo "The current path: ${CURRENT_PATH}"
echo "The wrapper path: ${WRAPPER_PATH}"
cd ${PROJECT_PATH}
source $WRAPPER_PATH
cd ${CURRENT_PATH}

max_epoch=200

declare -a StringArray=(
"python train_IIC_Twohead.py Config=../config/IICClusterMultiHead_MNIST.yaml Trainer.save_dir=reproducibility1 Trainer.max_epoch=${max_epoch}" \
"python train_IIC_Twohead.py Config=../config/IICClusterMultiHead_MNIST.yaml Trainer.save_dir=reproducibility2 Trainer.max_epoch=${max_epoch}" \
"python train_IIC_Twohead.py Config=../config/IICClusterMultiHead_MNIST.yaml Trainer.save_dir=reproducibility3 Trainer.max_epoch=${max_epoch}" \
"python train_IIC_Twohead.py Config=../config/IICClusterMultiHead_MNIST.yaml Trainer.save_dir=reproducibility4 Trainer.max_epoch=${max_epoch}" \

)
time=1

for cmd in "${StringArray[@]}"
do
#echo $cmd
$cmd
#wrapper "${time}" "${cmd}"
done