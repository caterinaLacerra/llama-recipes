export CUDA_VISIBLE_DEVICES=6,7
torchrun --nnodes 1 --nproc_per_node 2 llama_recipes/finetuning.py --dist_checkpoint_folder debug-new-features --data_path /home/caterina/sharegpt-data/improved-validation
