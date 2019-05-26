#!/usr/bin/env bash
python main.py Trainer.save_dir=MI_MNIST/use_vat Trainer.use_vat=True
python main.py Trainer.save_dir=MI_MNIST/use_rt Trainer.use_vat=False