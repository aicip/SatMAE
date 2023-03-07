#!/bin/bash

EPOCHS=200
BATCH_SIZE=512

INPUT_SIZE=64
PATCH_SIZE=8

FFN_NAME="MLP"
LR=0.0005

MODEL=$1
LOSS=$2
ATTENTION=$3

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model> <loss> <attention> [additional flags]"
    echo "Losses: mse, l1"
    echo "Attentions: scaled_dot_product, linformer, orthoformer, random, local, nystrom, fourier_mix"
    exit 1
fi
shift

if [ -z "$LOSS" ]; then
    echo "Usage: $0 <model> <loss> <attention> [additional flags]"
    echo "Losses: mse, l1"
    echo "Attentions: scaled_dot_product, linformer, orthoformer, random, local, nystrom, fourier_mix"
    exit 1
fi
shift

if [ -z "$ATTENTION" ]; then
    echo "Usage: $0 <model> <loss> <attention> [additional flags]"
    echo "Losses: mse, l1"
    echo "Attentions: scaled_dot_product, linformer, orthoformer, random, local, nystrom, fourier_mix"
    exit 1
fi
shift

IN_PATH_BASE="../fmow-rgb-preproc"
IN_PATH="${IN_PATH_BASE}/train_${INPUT_SIZE}.csv"
# IN_PATH="$IN_PATH_BASE/train_${INPUT_SIZE}_com2044.csv"

# OUT_DIR_BASE="."
OUT_DIR_BASE="../Model_Saving"
OUT_DIR="${OUT_DIR_BASE}/out_${MODEL}_xformers_${ATTENTION}_${FFN_NAME}_i${INPUT_SIZE}_p${PATCH_SIZE}_b${BATCH_SIZE}_e${EPOCHS}_${LOSS}_lr${LR}"

# Note: If you want to use additional flags, pass them when running the script.
# Example: ./trainX.sh --wandb satmae --device "cuda:0"

python3 main_pretrain.py --use-xformers \
    --train_path "$IN_PATH" \
    --output_dir "$OUT_DIR" \
    --model="$MODEL" \
    --loss "$LOSS" \
    --lr "$LR" \
    --attn_name "$ATTENTION" \
    --ffn_name="$FFN_NAME" \
    --input_size "$INPUT_SIZE" \
    --patch_size "$PATCH_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    $@
