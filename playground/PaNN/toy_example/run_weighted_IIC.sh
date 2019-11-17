#!/usr/bin/env bash
#python toy_example.py Trainer.save_dir=IIC/IIC_weighted Trainer.name=SemiWeightedIICTrainer
#python toy_example.py Trainer.save_Dir=IIC/conventional Trainer.name=SemiWeightedIICTrainer Trainer.use_prior=True

python toy_example.py Trainer.save_dir=UDA/UDA_convention Trainer.name=SemiUDATrainer
python toy_example.py Trainer.save_dir=UDA/UDA_weighted Trainer.name=SemiUDATrainer Trainer.use_prior=True