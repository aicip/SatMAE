#!/bin/bash

EPOCHS=200
BATCH_SIZE=512

INPUT_SIZE=64
PATCH_SIZE=8

FFN_NAME="MLP"
LR=0.0005

MODEL=$1
ATTENTION=$2
LOSS=$3

function usage {
    echo "Usage: $0 <model> <attention> <loss> [additional flags]"
    exit 1
}

# if any of the above are empty, then usage
if [ -z "$MODEL" ] || [ -z "$ATTENTION" ] || [ -z "$LOSS" ]; then
    usage
fi

shift 3

IN_PATH_BASE="../fmow-rgb-preproc"
IN_PATH="${IN_PATH_BASE}/train_${INPUT_SIZE}.csv"
# IN_PATH="$IN_PATH_BASE/train_${INPUT_SIZE}_com2044.csv"

# OUT_DIR_BASE="."
OUT_DIR_BASE="../Model_Saving"

# Note: If you want to use additional flags, pass them when running the script.
# Example: ./trainX.sh --wandb satmae --device "cuda:0"

python3 main_pretrain.py --use_xformers \
    --train_path "$IN_PATH" \
    --output_dir_base "$OUT_DIR_BASE" \
    --model="$MODEL" \
    --loss "$LOSS" \
    --lr "$LR" \
    --attn_name "$ATTENTION" \
    --ffn_name="$FFN_NAME" \
    --input_size "$INPUT_SIZE" \
    --patch_size "$PATCH_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" $@
