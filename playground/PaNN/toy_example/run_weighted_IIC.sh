#!/usr/bin/env bash
python toy_example.py Trainer.save_dir=IIC/IIC_weighted Trainer.name=SemiWeightedIICTrainer
python toy_example.py Trainer.save_Dir=IIC/conventional Trainer.name=SemiWeightedIICTrainer Trainer.use_prior=True