#!/usr/bin/env bash
python -O swa_main.py Config=config.yaml Trainer.max_epoch=2000 Arch.name=wideresnet Trainer.save_dir=cifar/benchmark/wideresnet
python -O swa_main.py Config=config.yaml Trainer.max_epoch=2000 Arch.name=largeconvnet Trainer.save_dir=cifar/benchmark/largeconvnet
