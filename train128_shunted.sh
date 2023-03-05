#!/bin/zsh

DEVICE="cuda:1"

EPOCHS=200

INPUT_SIZE=128
BATCH_SIZE=64
PATCH_SIZES="4|4"
EMBED_DIMS='64|128'
DEPTHS='1|2'
NUM_HEADS='8|16'
MLP_RATIOS='8|4'
SR_RATIOS='2|2'

PRINT_LEVEL=3
ATTENTION="shunted"
MODEL_NAME="shunted_mae_vit_large_patch16"

# Data path for com1822:
# IN_PATH_BASE="/data2/HDD_16TB"
# IN_PATH="${IN_PATH_BASE}/fmow-rgb-preproc/train_${INPUT_SIZE}.csv"
# Data path for com2044
IN_PATH_BASE="/mnt/com1822_HDD_16TB"
IN_PATH="${IN_PATH_BASE}/fmow-rgb-preproc/train_${INPUT_SIZE}_com2044.csv"

# Data path for com1822:
# OUT_DIR_BASE="/data2/HDD_16TB/ICCV/Model_Saving"
# Data path for com2044
OUT_DIR_BASE="/mnt/com1822_HDD_16TB/ICCV/Model_Saving"
OUT_DIR="${OUT_DIR_BASE}/out_i${INPUT_SIZE}_p${PATCH_SIZES}_e${EMBED_DIMS}_d${DEPTHS}_h${NUM_HEADS}_mlp${MLP_RATIOS}_sr${SR_RATIOS}_e${EPOCHS}_${ATTENTION}"

WANDB="satmae"

python3 main_pretrain.py \
--device "${DEVICE}" \
--train_path "${IN_PATH}" \
--output_dir "${OUT_DIR}" \
--input_size "${INPUT_SIZE}" \
--patch_sizes "${PATCH_SIZES}" \
--embed_dims "${EMBED_DIMS}" \
--depths "${DEPTHS}" \
--num_heads "${NUM_HEADS}" \
--mlp_ratios "${MLP_RATIOS}" \
--sr_ratios "${SR_RATIOS}" \
--batch_size "${BATCH_SIZE}" \
--epochs "${EPOCHS}" \
--attention "${ATTENTION}" \
--print_level "${PRINT_LEVEL}" \
--model "${MODEL_NAME}" \
--wandb "${WANDB}" $@
