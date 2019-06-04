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

# IMSAT
# 1. IMSATMI(x,p) + CE(p,adv(p))  -->orignal IMSAT okay
# 2. MI(x,p) + CE(p,geom(p))   done
# 3. MI(x,p) + CE(p,adv(p)) + CE(p,geom(p))   haven't program this , you want to mix them up okay
# IIC    
# 4. MI(p,geom(p)) --> original IIC done    
# 5. MI(p,adv(p)) --> I haven't program this      
# 6. MI(p,geom(p)) + MI(p,adv(p)) --> done

weight=0.01
declare -a StringArray=(
"python main.py Trainer.save_dir=MI_MNIST/IMSAT/case1_${weight} Trainer.name=IMSAT Trainer.use_vat=True Trainer.sat_weight=${weight} \
#Trainer.checkpoint_path=${PROJECT_PATH}/runs/MI_MNIST/IMSAT/case1_${weight}" \
"python main.py Trainer.save_dir=MI_MNIST/IMSAT/case2_${weight} Trainer.name=IMSAT Trainer.use_vat=False Trainer.sat_weight=${weight} \
#Trainer.checkpoint_path=${PROJECT_PATH}/runs/MI_MNIST/IMSAT/case2_${weight}" \
"python main.py Trainer.save_dir=MI_MNIST/IMSAT/case3_${weight} Trainer.name=IMSAT_enhance Trainer.sat_weight=${weight} \
#Trainer.checkpoint_path=${PROJECT_PATH}/runs/MI_MNIST/IMSAT/case3_${weight}" \
\
"python main.py Trainer.save_dir=MI_MNIST/IIC/case4 Trainer.name=IIC Trainer.sat_weight=0 #Trainer.checkpoint_path=${PROJECT_PATH}/runs/MI_MNIST/IIC/case4" \
"python main.py Trainer.save_dir=MI_MNIST/IIC/case5 Trainer.name=IIC_adv_enhance #Trainer.checkpoint_path=${PROJECT_PATH}/runs/MI_MNIST/IIC/case5" \
"python main.py Trainer.save_dir=MI_MNIST/IIC/case6 Trainer.name=IIC_enhance Trainer.sat_weight=0 #Trainer.checkpoint_path=${PROJECT_PATH}/runs/MI_MNIST/IIC/case6" \
)
time=1

for cmd in "${StringArray[@]}"
do
echo $cmd
wrapper "${time}" "${cmd}"
done



#
