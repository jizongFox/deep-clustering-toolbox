#!/usr/bin/env bash
# this script is provided to compare the performance between iic and imsat using different MI and data augmentation
max_epoch=1
# only using IIC:
python train_IIC_IMSAT.py Trainer.save_dir=onlyIIC Trainer.IIC_weight=1.0 Trainer.IMSAT_weight=0.0 Trainer.max_epoch=$max_epoch
# only using IMSAT:
python train_IIC_IMSAT.py Trainer.save_dir=onlyIMSAT Trainer.IIC_weight=0.0 Trainer.IMSAT_weight=1.0 Trainer.max_epoch=$max_epoch
# mixed them:
python train_IIC_IMSAT.py Trainer.save_dir=ratio_0.01 Trainer.IIC_weight=1 Trainer.IMSAT_weight=0.01 Trainer.max_epoch=$max_epoch

python train_IIC_IMSAT.py Trainer.save_dir=ratio_0.1 Trainer.IIC_weight=1 Trainer.IMSAT_weight=0.1 Trainer.max_epoch=$max_epoch

python train_IIC_IMSAT.py Trainer.save_dir=ratio_1 Trainer.IIC_weight=1 Trainer.IMSAT_weight=1 Trainer.max_epoch=$max_epoch

python train_IIC_IMSAT.py Trainer.save_dir=ratio_10 Trainer.IIC_weight=1 Trainer.IMSAT_weight=10 Trainer.max_epoch=$max_epoch