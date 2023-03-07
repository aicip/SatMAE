#!/bin/bash

EPOCHS=200
BATCH_SIZE=512

INPUT_SIZE=64
PATCH_SIZE=8

MODEL="mae_vit_tiny"
FFN_NAME="MLP"
LOSS="l1"
LR=0.0005

ATTENTION="scaled_dot_product"

IN_PATH_BASE="../fmow-rgb-preproc"
IN_PATH="$IN_PATH_BASE/train_${INPUT_SIZE}.csv"
# IN_PATH="$IN_PATH_BASE/train_${INPUT_SIZE}_com2044.csv"

# OUT_DIR_BASE="."
OUT_DIR_BASE="../Model_Saving"
OUT_DIR="${OUT_DIR_BASE}/out_${MODEL}_xformers_${ATTENTION}_${FFN_NAME}_i${INPUT_SIZE}_p${PATCH_SIZE}_b${BATCH_SIZE}_e${EPOCHS}_${LOSS}_lr${LR}"

# Note: If you want to use additional flags, pass them when running the script.
# Example: ./trainX.sh --wandb satmae --device "cuda:0"

python3 main_pretrain.py \
    --train_path "$IN_PATH" \
    --output_dir "$OUT_DIR" \
    --model="$MODEL" \
    --input_size "$INPUT_SIZE" \
    --patch_size "$PATCH_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --loss "$LOSS" \
    --lr "$LR" \
    --attn_name "$ATTENTION" \
    --ffn_name="$FFN_NAME" $@