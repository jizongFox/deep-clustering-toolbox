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
"python main.py Trainer.save_dir=IIC_VAT/baseline" \
"python main.py Trainer.save_dir=IIC_VAT/adv_0.01 Trainer.adv_weight=0.01" \
"python main.py Trainer.save_dir=IIC_VAT/adv_0.1 Trainer.adv_weight=0.1" \
"python main.py Trainer.save_dir=IIC_VAT/adv_1.0 Trainer.adv_weight=1.00" \
)
time=1

for cmd in "${StringArray[@]}"
do
echo $cmd
wrapper "${time}" "${cmd}"
done



#
