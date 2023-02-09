#!/bin/zsh

python3 main_pretrain.py --device cuda:0 --train_path /data2/HDD_16TB/fmow-rgb-preproc/train_112.csv --output_dir out112_b64_e200 --input_size 112 --batch_size 64
