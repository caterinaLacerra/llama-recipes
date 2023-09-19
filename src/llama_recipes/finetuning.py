# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from pathlib import Path

from pkg_resources import packaging

import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    default_data_collator
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

from llama_recipes.configs import FSDPConfig, TrainConfig
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.train_utils import (
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)
from llama_recipes.utils.training_loop import train

def is_lama_model(train_config: TrainConfig) -> bool:
    return 'llama' in train_config.model_name.lower()


def has_collate_fn(dataset_instance: Dataset) -> bool:
    collate = getattr(dataset_instance, "collate_fn", None)
    return callable(collate)


def main(**kwargs):
    # Update the configuration for the training and sharding process
    update_config((TrainConfig, FSDPConfig), **kwargs)
    save_path = Path(TrainConfig.dist_checkpoint_root_folder).joinpath(TrainConfig.dist_checkpoint_folder)
    if TrainConfig.save_model and save_path.exists() and save_path.is_dir() and list(save_path.iterdir()) != []:
        raise ValueError(f"Please define a new folder to save checkpoints: {save_path} exists and is not empty.")

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(TrainConfig.seed)
    torch.manual_seed(TrainConfig.seed)

    if TrainConfig.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load the pre-trained model and setup its configuration
    use_cache = False if TrainConfig.enable_fsdp else None
    if TrainConfig.enable_fsdp and TrainConfig.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = AutoModelForCausalLM.from_pretrained(
                TrainConfig.model_name,
                load_in_8bit=True if TrainConfig.quantization else None,
                device_map="auto" if TrainConfig.quantization else None,
                use_cache=use_cache,
            )
        else:
            llama_config = AutoConfig.from_pretrained(TrainConfig.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = AutoModelForCausalLM(llama_config)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            TrainConfig.model_name,
            load_in_8bit=True if TrainConfig.quantization else None,
            device_map="auto" if TrainConfig.quantization else None,
            use_cache=use_cache,
        )
    if TrainConfig.enable_fsdp and TrainConfig.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model) 
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(model, TrainConfig, rank if TrainConfig.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if TrainConfig.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if TrainConfig.enable_fsdp and FSDPConfig.pure_bf16:
        model.to(torch.bfloat16)

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(TrainConfig.model_name)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
    if TrainConfig.use_peft:
        peft_config = generate_peft_config(TrainConfig, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    #setting up FSDP if enable_fsdp is enabled
    if TrainConfig.enable_fsdp:
        if not TrainConfig.use_peft and TrainConfig.freeze_layers:

            freeze_transformer_layers(TrainConfig.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(FSDPConfig, rank)

        if is_lama_model(TrainConfig):
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
        else:
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, GPTNeoXLayer)

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if TrainConfig.use_peft else wrapping_policy,
            mixed_precision=mixed_precision_policy if not FSDPConfig.pure_bf16 else None,
            sharding_strategy=FSDPConfig.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=TrainConfig.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if TrainConfig.low_cpu_fsdp and rank != 0 else None,
        )
        if FSDPConfig.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not TrainConfig.quantization and not TrainConfig.enable_fsdp:
        model.to("cuda")

    dataset_config = generate_dataset_config(TrainConfig, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not TrainConfig.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if not TrainConfig.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    train_sampler = None
    val_sampler = None
    if TrainConfig.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if TrainConfig.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )

    if has_collate_fn(dataset_train):
        collator = default_data_collator
    else:
        collator = dataset_train.collate_fn
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=TrainConfig.batch_size_training,
        num_workers=TrainConfig.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=False,
        collate_fn=collator,
    )

    eval_dataloader = None
    if TrainConfig.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=TrainConfig.val_batch_size,
            num_workers=TrainConfig.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=False,
            collate_fn=collator,
        )

    # Initialize the optimizer and learning rate scheduler
    if FSDPConfig.pure_bf16 and FSDPConfig.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=TrainConfig.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=TrainConfig.lr,
            weight_decay=0.0,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=TrainConfig.gamma)

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        TrainConfig.gradient_accumulation_steps,
        TrainConfig,
        FSDPConfig if TrainConfig.enable_fsdp else None,
        local_rank if TrainConfig.enable_fsdp else None,
        rank if TrainConfig.enable_fsdp else None,
    )
    if not TrainConfig.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)
