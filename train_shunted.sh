#!/bin/bash

##################### Defaults #####################
EPOCHS=200
INPUT_SIZE=64
PATCH_SIZES="4+4"
ATTENTION="shunted"
OUT_DIR_BASE="../Model_Saving"
WANDB_PROJECT="satmae"
START_EPOCH=0  # Modify below
RESUME=""  # Modify below
WANDB_ID=""  # Modify below
# Mofify in path based on hostname
IN_PATH_BASE="../fmow-rgb-preproc"
IN_PATH="${IN_PATH_BASE}/train_${INPUT_SIZE}"
if ! hostname | grep -q "^com1822"; then
  IN_PATH="${IN_PATH}_com2044"
fi
IN_PATH="${IN_PATH}.csv"
####################################################

#################### Parameters ####################
PRINT_LEVEL=1
# ------- Model Name ------- #
MODEL_NAME="mae_vit_tiny_shunted_2st"
# MODEL_NAME="mae_vit_mini_shunted_2st"
# MODEL_NAME="mae_vit_small_shunted_2st"
# MODEL_NAME="mae_vit_base_shunted_2st"
# MODEL_NAME="mae_vit_tiny_shunted_2st_cross"
# MODEL_NAME="mae_vit_small_shunted_2st_cross"

# ------- Run Specific ------- #
DEVICE="cuda:3"
BATCH_SIZE=512
# Resume from checkpoint
# CHECKPOINT_DIR="${OUT_DIR_BASE}/to-be-filled"
# START_EPOCH=0
# RESUME="${RESUME}/checkpoint-${START_EPOCH}.pth"
# WANDB_ID="r3dr4f4"  # take his from the run url

# ------- Hyperparams ------- #
LOSS="l1_full"
LR=0.001
# LOSS="mse_full"
# LR=0.0001
MASK_RATIO='0.75'
# MASK_RATIO='0.50'
####################################################

python3 main_pretrain.py \
    --train_path "${IN_PATH}" \
    --output_dir_base "${OUT_DIR_BASE}" \
    --model "${MODEL_NAME}" \
    --loss "${LOSS}" \
    --lr "${LR}" \
    --attn_name "${ATTENTION}" \
    --input_size "${INPUT_SIZE}" \
    --patch_size "${PATCH_SIZES}" \
    --mask_ratio "${MASK_RATIO}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --device "${DEVICE}" \
    --print_level "${PRINT_LEVEL}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_id "${MODEL_NAME}" \
    --resume "${RESUME}" \
    --start_epoch "${START_EPOCH}" $@
 
