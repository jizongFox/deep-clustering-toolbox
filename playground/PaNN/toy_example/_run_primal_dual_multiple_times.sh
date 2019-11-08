#!/usr/bin/env bash

CC_WRAPPER_PATH=$(python -c "import deepclustering; print(deepclustering.CC_wrapper_path)")
LC_WRAPPER_PATH=$(python -c "import deepclustering; print(deepclustering.LC_wrapper_path)")
JA_WRAPPER_PATH=$(python -c "import deepclustering; print(deepclustering.JA_wrapper_path)")
source $CC_WRAPPER_PATH # enable wrapper
source $LC_WRAPPER_PATH # enable local_wrapper
source $JA_WRAPPER_PATH
seed=$1

save_dir=pann/primal-dual_update_mu_by_RADAM/seed_${seed}
max_epoch=200
time=8
account=def-chdesa
FORCE_LOAD_CHECKPOINT=1

declare -a StringArray=(
"FORCE_LOAD_CHECKPOINT=${FORCE_LOAD_CHECKPOINT} python   toy_example.py  Trainer.save_dir=${save_dir}/baseline  Trainer.name=SemiTrainer Trainer.max_epoch=${max_epoch} Seed=${seed} "
"FORCE_LOAD_CHECKPOINT=${FORCE_LOAD_CHECKPOINT} python   toy_example.py  Trainer.save_dir=${save_dir}/Entropy  Trainer.name=SemiEntropyTrainer Trainer.max_epoch=${max_epoch} Seed=${seed}  "
"FORCE_LOAD_CHECKPOINT=${FORCE_LOAD_CHECKPOINT} python   toy_example.py  Trainer.save_dir=${save_dir}/Entropy_CEntropy  Trainer.name=SemiEntropyTrainer Trainer.use_centropy=True Trainer.max_epoch=${max_epoch} Seed=${seed}  "
"FORCE_LOAD_CHECKPOINT=${FORCE_LOAD_CHECKPOINT} python   toy_example.py  Trainer.save_dir=${save_dir}/PrimalDual  Trainer.name=SemiPrimalDualTrainer Trainer.max_epoch=${max_epoch} Seed=${seed}  "
"FORCE_LOAD_CHECKPOINT=${FORCE_LOAD_CHECKPOINT} python   toy_example.py  Trainer.save_dir=${save_dir}/PrimalDual_CEntropy  Trainer.name=SemiPrimalDualTrainer Trainer.use_centropy=True Trainer.max_epoch=${max_epoch} Seed=${seed}  "

)
gpuqueue "${StringArray[@]}" --available_gpus 0

#for cmd in "${StringArray[@]}"
#do
#echo ${cmd}
##CC_wrapper "${time}" "${account}" "${cmd}" 16
##JA_wrapper "${time}" "${account}" "${cmd}" 16 4
#local_wrapper "${cmd}"
## ${cmd}
#done
