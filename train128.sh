#!/bin/zsh

DEVICE="cuda:0"

EPOCHS=200

INPUT_SIZE=128
PATCH_SIZE=16
BATCH_SIZE=64

ATTENTION="scaled_dot_product"

# IN_PATH_BASE="$HOME/Documents/datasets/image/fmow-rgb"
IN_PATH_BASE="/data2/HDD_16TB/fmow-rgb-preproc"
IN_PATH="$IN_PATH_BASE/train_$INPUT_SIZE.csv"

# OUT_DIR_BASE="."
OUT_DIR_BASE="/data2/HDD_16TB/ICCV/Model_Saving"
OUT_DIR="${OUT_DIR_BASE}/out_i${INPUT_SIZE}_p${PATCH_SIZE}_b${BATCH_SIZE}_e${EPOCHS}_${ATTENTION}"

python3 main_pretrain.py --device "$DEVICE" --train_path "$IN_PATH" --output_dir "$OUT_DIR" --input_size "$INPUT_SIZE" --patch_size "$PATCH_SIZE" --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --attention "$ATTENTION"
