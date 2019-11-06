#!/usr/bin/env bash
save_dir=pann/check_optimizer
max_epoch=350
declare -a StringArray=(
"python -OO  main.py Optim.name=RAdam      Trainer.save_dir=${save_dir}/RADAM      Trainer.max_epoch=${max_epoch}"
"python -OO  main.py Optim.name=Adam       Trainer.save_dir=${save_dir}/ADAM       Trainer.max_epoch=${max_epoch}"
"python -OO  main.py Optim.name=AdaBound   Trainer.save_dir=${save_dir}/ADABOUD    Trainer.max_epoch=${max_epoch}"
"python -OO  main.py Optim.name=AdaBoundW   Trainer.save_dir=${save_dir}/ADABOUDW  Trainer.max_epoch=${max_epoch}"
)
gpuqueue "${StringArray[@]}" --available_gpus 0 1