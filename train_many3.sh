./train64_small_xformers_l1.sh scaled_dot_product --wandb satmae
./train64_small_xformers_mse.sh scaled_dot_product --wandb satmae

./train64_small_xformers_mse.sh linformer --wandb satmae
./train64_small_xformers_mse.sh local --wandb satmae
./train64_small_xformers_mse.sh fourier_mix --wandb satmae
./train64_small_xformers_mse.sh orthoformer --wandb satmae

./train64_small_xformers_l1.sh linformer --wandb satmae
./train64_small_xformers_l1.sh local --wandb satmae
./train64_small_xformers_l1.sh fourier_mix --wandb satmae
./train64_small_xformers_l1.sh orthoformer --wandb satmae
