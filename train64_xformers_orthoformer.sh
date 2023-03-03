#!/bin/zsh

DEVICE="cuda:0"

EPOCHS=200
BATCH_SIZE=128

INPUT_SIZE=64
PATCH_SIZE=8

MODEL="mae_vit_small"
FFN_NAME="MLP"
ATTENTION="orthoformer"

IN_PATH_BASE="../fmow-rgb-preproc"
IN_PATH="$IN_PATH_BASE/train_$INPUT_SIZE.csv"

# OUT_DIR_BASE="."
OUT_DIR_BASE="../Model_Saving"
OUT_DIR="${OUT_DIR_BASE}/out_${MODEL}_xformers_${ATTENTION}_${FFN_NAME}_i${INPUT_SIZE}_p${PATCH_SIZE}_b${BATCH_SIZE}_e${EPOCHS}"

# WANDB="satmae"
WANDB="satmae_testing"

python3 main_pretrain.py --use-xformers --device "$DEVICE" --train_path "$IN_PATH" --output_dir "$OUT_DIR" --model="$MODEL" --input_size "$INPUT_SIZE" --patch_size "$PATCH_SIZE" --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --attn_name "$ATTENTION" --ffn_name="$FFN_NAME" --wandb "$WANDB" $@