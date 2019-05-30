#!/usr/bin/env bash
#python mnist.py loss=mse Trainer.save_dir=mnist_autoencoder/baseline &
python mnist.py loss=gdl weight=0.00 Trainer.save_dir=mnist_autoencoder/baseline  &
python mnist.py loss=gdl weight=0.01 Trainer.save_dir=mnist_autoencoder/gdl_0.01  &
python mnist.py loss=gdl weight=0.1 Trainer.save_dir=mnist_autoencoder/gdl_0.1  &
python mnist.py loss=gdl weight=1 Trainer.save_dir=mnist_autoencoder/gdl_1.0  &