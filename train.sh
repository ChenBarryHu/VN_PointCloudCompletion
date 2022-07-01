#!/usr/bin/env sh
python train.py --exp_name train_coarse --lr 0.0001 --epochs 400 --batch_size 32 --coarse_loss cd --num_workers 8