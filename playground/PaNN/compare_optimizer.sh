#!/usr/bin/env bash

CC_WRAPPER_PATH=$(python -c "import deepclustering; print(deepclustering.CC_wrapper_path)")
LC_WRAPPER_PATH=$(python -c "import deepclustering; print(deepclustering.LC_wrapper_path)")
JA_WRAPPER_PATH=$(python -c "import deepclustering; print(deepclustering.JA_wrapper_path)")
source $CC_WRAPPER_PATH # enable wrapper
source $LC_WRAPPER_PATH # enable local_wrapper
source $JA_WRAPPER_PATH

save_dir=pann/check_optimizer
max_epoch=350
time=8
account=def-chdesa
FORCE_LOAD_CHECKPOINT=1

declare -a StringArray=(
"FORCE_LOAD_CHECKPOINT=${FORCE_LOAD_CHECKPOINT} python -OO  main.py Optim.name=RAdam      Trainer.save_dir=${save_dir}/RADAM      Trainer.max_epoch=${max_epoch} Trainer.checkpoint_path=../../runs/${save_dir}/RADAM "
"FORCE_LOAD_CHECKPOINT=${FORCE_LOAD_CHECKPOINT} python -OO  main.py Optim.name=Adam       Trainer.save_dir=${save_dir}/ADAM       Trainer.max_epoch=${max_epoch} Trainer.checkpoint_path=../../runs/${save_dir}/ADAM "
"FORCE_LOAD_CHECKPOINT=${FORCE_LOAD_CHECKPOINT} python -OO  main.py Optim.name=AdaBound   Trainer.save_dir=${save_dir}/ADABOUD    Trainer.max_epoch=${max_epoch} Trainer.checkpoint_path=../../runs/${save_dir}/ADABOUD "
"FORCE_LOAD_CHECKPOINT=${FORCE_LOAD_CHECKPOINT} python -OO  main.py Optim.name=AdaBoundW   Trainer.save_dir=${save_dir}/ADABOUDW  Trainer.max_epoch=${max_epoch} Trainer.checkpoint_path=../../runs/${save_dir}/ADABOUDW "
"FORCE_LOAD_CHECKPOINT=${FORCE_LOAD_CHECKPOINT} python -OO  main.py Optim.name=SGD Optim.momentum=0.9   Trainer.save_dir=${save_dir}/SGD_m  Trainer.max_epoch=${max_epoch} Trainer.checkpoint_path=../../runs/${save_dir}/SGD_m "
"FORCE_LOAD_CHECKPOINT=${FORCE_LOAD_CHECKPOINT} python -OO  main.py Optim.name=SGD Trainer.save_dir=${save_dir}/SGD  Trainer.max_epoch=${max_epoch} Trainer.checkpoint_path=../../runs/${save_dir}/SGD "
)
#gpuqueue "${StringArray[@]}" --available_gpus 0

for cmd in "${StringArray[@]}"
do
echo ${cmd}
CC_wrapper "${time}" "${account}" "${cmd}" 16
#JA_wrapper "${time}" "${account}" "${cmd}" 16 4
#local_wrapper "${cmd}"
# ${cmd}
done
