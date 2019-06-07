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



declare -a StringArray=(
"python train_IIC_Twohead.py Config=../config/IICClusterMultiHead_CIFAR.yaml Trainer.save_dir=test_pipeline/cifar10 Trainer.max_epoch=100" \
"python train_IIC_Twohead.py Config=../config/IICClusterMultiHead_MNIST.yaml Trainer.save_dir=test_pipeline/mnist Trainer.max_epoch=100" \
"python train_IIC_Twohead.py Config=../config/IICClusterMultiHead_STL10.yaml Trainer.save_dir=test_pipeline/stl10 Trainer.max_epoch=100" \
)
time=1

for cmd in "${StringArray[@]}"
do
#echo $cmd
$cmd
#wrapper "${time}" "${cmd}"
done
