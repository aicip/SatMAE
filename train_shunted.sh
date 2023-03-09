#!/bin/bash
# !/bin/zsh

DEVICE="cuda:1"

EPOCHS=200

INPUT_SIZE=64
BATCH_SIZE=128

PATCH_SIZES="4-4"
MASK_RATIO='0.75'
# MASK_RATIO='0.60'
# LR=0.0001
# LOSS="mse"
LR=0.001
LOSS="l1"

PRINT_LEVEL=1
ATTENTION="shunted"

# MODEL_NAME="shunted_2s_mae_vit_tiny"
# MODEL_NAME="shunted_2s_mae_vit_mini"
# MODEL_NAME="mae_vit_small_shunted_2st"
# MODEL_NAME="shunted_2s_mae_vit_base"
MODEL_NAME="shunted_2s_mae_vit_tiny_cross"
# MODEL_NAME="mae_vit_small_shunted_2st_cross"

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

OUT_DIR="${OUT_DIR_BASE}/out_${MODEL_NAME}_i${INPUT_SIZE}_p${PATCH_SIZES}_e${EPOCHS}_mask${MASK_RATIO}_${LOSS}_lr${LR}"

# RESUME="${OUT_DIR_BASE}/out_mae_vit_tiny_shunted_2st_cross_i64_p4-4_e200_mask0.75_l1_lr0.001"
# START_EPOCH=100
# RESUME="${RESUME}/checkpoint-${START_EPOCH}.pth"
# OUT_DIR="${OUT_DIR}_resume"

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
--loss "${LOSS}" \
--mask_ratio "${MASK_RATIO}" \
--wandb "${WANDB}" \
--resume "${RESUME}" \
--start_epoch "${START_EPOCH}" \
 
