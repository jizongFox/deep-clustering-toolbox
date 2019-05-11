#!/usr/bin/env bash
python script/train_IIC_Twohead.py Trainer.save_dir=runs/2head_1subhead_k_30 Arch.num_sub_heads=5 Arch.output_k_A=30
