#!/bin/zsh

DEVICE="cuda:1"

EPOCHS=200

INPUT_SIZE=64
BATCH_SIZE=256tm

PATCH_SIZES="4|4"
# MASK_RATIO='0.60'
MASK_RATIO='0.60'
LR=0.0001

PRINT_LEVEL=1
ATTENTION="shunted"

MODEL_NAME="shunted_2s_mae_vit_tiny"
# MODEL_NAME="shunted_2s_mae_vit_mini"
# MODEL_NAME="shunted_2s_mae_vit_small"
# MODEL_NAME="shunted_2s_mae_vit_base"

# Data path for com1822:
IN_PATH_BASE="/data2/HDD_16TB"
IN_PATH="${IN_PATH_BASE}/fmow-rgb-preproc/train_${INPUT_SIZE}.csv"
# Data path for com2044
# IN_PATH_BASE="/mnt/com1822_HDD_16TB"
# IN_PATH="${IN_PATH_BASE}/fmow-rgb-preproc/train_${INPUT_SIZE}_com2044.csv"

# Data path for com1822:
OUT_DIR_BASE="/data2/HDD_16TB/ICCV/Model_Saving"
# Data path for com2044
# OUT_DIR_BASE="/mnt/com1822_HDD_16TB/ICCV/Model_Saving"

OUT_DIR="${OUT_DIR_BASE}/out_${MODEL}_i${INPUT_SIZE}_p${PATCH_SIZES}_e${EPOCHS}_${ATTENTION}_ratio${MASK_RATIO}_lr${LR}"

WANDB="satmae"

python3 main_pretrain.py \
--device "${DEVICE}" \
--train_path "${IN_PATH}" \
--output_dir "${OUT_DIR}" \
--input_size "${INPUT_SIZE}" \
--patch_size "${PATCH_SIZES}" \
--batch_size "${BATCH_SIZE}" \
--epochs "${EPOCHS}" \
--attn_name "${ATTENTION}" \
--print_level "${PRINT_LEVEL}" \
--model "${MODEL_NAME}" \
--lr "${LR}" \
--mask_ratio "${MASK_RATIO}" \
--wandb "${WANDB}" $@