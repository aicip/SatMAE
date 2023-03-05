# time: 0.0700 max mem: 2095
./train64_xformers.sh scaled_dot_product --wandb satmae_debug --loss l1

# time: 0.2427 max mem: 2056
./train64_xformers.sh orthoformer --wandb satmae_debug

# time: 0.0530 max mem: 2122
./train64_xformers.sh linformer --wandb satmae_debug

# FAILED
./train64_xformers.sh fourier_mix --wandb satmae_debug

# time: 0.1433 max mem: 3177
# ./train64_xformers.sh favor --wandb satmae_debug

# FAILED
# ./train64_xformers.sh nystrom --wandb satmae_debug

# time: 0.0733 max mem: 2134
# ./train64_xformers.sh random --wandb satmae_debug

# time: 0.0595 max mem: 2183
# ./train64_xformers.sh local --wandb satmae_debug
