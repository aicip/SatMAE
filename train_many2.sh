# ./train64_xformers.sh scaled_dot_product --wandb satmae_debug --loss l1
# ./train64_xformers.sh orthoformer --wandb satmae_debug
# ./train64_xformers.sh linformer --wandb satmae_debug
# ./train64_xformers.sh fourier_mix --wandb satmae_debug

./train64_xformers.sh favor --wandb satmae_debug
./train64_xformers.sh nystrom --wandb satmae_debug
./train64_xformers.sh random --wandb satmae_debug
./train64_xformers.sh local --wandb satmae_debug
