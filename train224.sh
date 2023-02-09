#!/bin/zsh

python3 main_pretrain.py --device cuda:3 --train_path /data2/HDD_16TB/fmow-rgb-preproc/train_224.csv --output_dir out224_b64_e200 --input_size 224 --batch_size 64
