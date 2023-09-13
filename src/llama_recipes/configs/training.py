# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="/data/models/pythia-70m-deduped"
    enable_fsdp: bool=True
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=4
    gradient_accumulation_steps: int=1
    validate_every_n_steps: int=50 # perform validation and save checkpoints every N steps (if -1, waits for the end of the epoch)
    num_epochs: int=100
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=True
    mixed_precision: bool=False # not supported
    val_batch_size: int=1
    dataset = "sharegpt_dataset"
    peft_method: str = "None" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="/data/caterina/ft-checkpoints/" # will be used if using FSDP
    dist_checkpoint_folder: str="" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    early_stopping: bool = True
    patience: int = 5 # patience for early stopping callback
    
    
    
