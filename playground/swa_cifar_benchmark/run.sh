#!/usr/bin/env bash
PYDEBUG=1 python -O swa_main.py Config=config.yaml Trainer.max_epoch=10
PYDEBUG=1 python -O swa_main.py Config=config_swa.yaml Trainer.max_epoch=10