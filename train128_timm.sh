#!/bin/zsh

DEVICE="cuda:0"

EPOCHS=200
BATCH_SIZE=256

INPUT_SIZE=128
PATCH_SIZE=16

MODEL="mae_vit_base"
FFN_NAME="MLP"
ATTENTION="scaled_dot_product"

IN_PATH_BASE="../fmow-rgb-preproc"
IN_PATH="$IN_PATH_BASE/train_$INPUT_SIZE.csv"

# OUT_DIR_BASE="."
OUT_DIR_BASE="../Model_Saving"
OUT_DIR="${OUT_DIR_BASE}/out_${MODEL}_timm_${ATTENTION}_${FFN_NAME}_i${INPUT_SIZE}_p${PATCH_SIZE}_b${BATCH_SIZE}_e${EPOCHS}"

# Note: If you want to use additional flags, pass them when running the script.
# Example: ./trainX.sh --wandb satmae

python3 main_pretrain.py --device "$DEVICE" --train_path "$IN_PATH" --output_dir "$OUT_DIR" --model="$MODEL" --input_size "$INPUT_SIZE" --patch_size "$PATCH_SIZE" --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --attn_name "$ATTENTION" --ffn_name="$FFN_NAME" $@
