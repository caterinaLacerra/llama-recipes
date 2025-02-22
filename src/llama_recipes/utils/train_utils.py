# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import dataclasses
import os
import time
from typing import Any, Dict, List

import transformers
import yaml
from pathlib import Path
from pkg_resources import packaging

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from tqdm import tqdm
from transformers import LlamaTokenizer

from llama_recipes.configs import TrainConfig, FSDPConfig
from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, \
    save_optimizer_checkpoint
from llama_recipes.policies import fpSixteen, bfSixteen_mixed, get_llama_wrapper


def run_validation(
        model: transformers.AutoModel.from_pretrained,
        tokenizer: transformers.AutoTokenizer.from_pretrained,
        optimizer: torch.optim.Optimizer,
        train_config: TrainConfig,
        fsdp_config: FSDPConfig,
        eval_dataloader: torch.utils.data.DataLoader,
        local_rank: int,
        rank: int,
        epoch: int,
        step: int,
        best_val_loss: torch.Tensor,
        val_loss: torch.Tensor,
        val_prep: torch.Tensor,
        checkpoint_times: List[float],
        current_patience: int
):
    eval_ppl, eval_epoch_loss = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer)
    checkpoint_start_time = time.perf_counter()
    if train_config.save_model and eval_epoch_loss < best_val_loss:
        if train_config.enable_fsdp:
            dist.barrier()
        if train_config.use_peft:
            distributed_print(message="We are about to save the PEFT modules", rank=rank, fsdp=train_config.enable_fsdp)
            model.save_pretrained(train_config.output_dir)
            distributed_print(message=f"PEFT modules are saved in {train_config.output_dir}", rank=rank,
                              fsdp=train_config.enable_fsdp)

        else:
            if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                save_model_checkpoint(
                    model, rank, train_config, epoch=epoch, step=step
                )
            elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                print("=====================================================")
                save_model_and_optimizer_sharded(model, rank, train_config)
                if train_config.save_optimizer:
                    save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                    print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                    print("=====================================================")

            if not train_config.use_peft and train_config.save_optimizer:
                save_optimizer_checkpoint(
                    model, optimizer, rank, train_config, epoch=epoch
                )
                print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                print("=====================================================")

        if train_config.enable_fsdp:
            dist.barrier()

    checkpoint_end_time = time.perf_counter() - checkpoint_start_time
    checkpoint_times.append(checkpoint_end_time)
    previous_loss = best_val_loss
    if eval_epoch_loss < best_val_loss:
        best_val_loss = eval_epoch_loss
        distributed_print(
            message=f"Updating loss. Best eval loss on epoch {epoch}, step {step} is {best_val_loss}, "
                    f"previous loss: {previous_loss}",
            rank=rank,
            fsdp=train_config.enable_fsdp
        )
        # reset patience
        current_patience = train_config.patience
    else:  # loss didn't improve
        current_patience -= 1

    val_loss.append(best_val_loss)
    val_prep.append(eval_ppl)
    distributed_print(
        f"Current validation loss: {best_val_loss}. Current patience: {current_patience}",
        rank=rank,
        fsdp=train_config.enable_fsdp
    )
    should_stop = False
    if current_patience == 0 and train_config.early_stopping:
        message = f"Reached best loss at {best_val_loss}. Stopping training for early stopping at epoch {epoch}"
        distributed_print(message, rank, train_config.enable_fsdp)
        should_stop = True

    return current_patience, checkpoint_times, val_loss, val_prep, best_val_loss, should_stop


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2 ** 20)


def gradient_update(train_config, scaler, gradient_accumulation_steps, train_dataloader, optimizer, pbar, loss, step):
    if train_config.use_fp16:
        # if fp16 is enabled, use gradient scaler to handle gradient update
        scaler.scale(loss).backward()
        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            pbar.update(step // gradient_accumulation_steps)
    else:
        # regular backpropagation when fp16 is not used
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(step // gradient_accumulation_steps)


def update_results(results, epoch_times, checkpoint_times, train_prep, train_loss, val_prep, val_loss, train_config):
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    return results


def distributed_print(message: str, rank: int, fsdp: bool) -> None:
    if fsdp:
        if rank == 0:
            print(message)
    else:
        print(message)


def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader
    
    Args:
        model: The model to evaluate
        train_config: Training configuration
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions
    
    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating")):
        for key in batch.keys():
            if train_config.enable_fsdp:
                batch[key] = batch[key].to(local_rank)
            else:
                batch[key] = batch[key].to('cuda:0')
        # Ensure no gradients are computed for this scope to save memory
        with torch.no_grad():
            # Forward pass and compute loss
            outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
        # Decode predictions and add to evaluation predictions list
        preds = torch.argmax(outputs.logits, -1)
        eval_preds.extend(
            tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
        )

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss / world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    distributed_print(f"{eval_ppl=} {eval_epoch_loss=}", rank=local_rank, fsdp=train_config.enable_fsdp)
    return eval_ppl, eval_epoch_loss


def freeze_transformer_layers(model, num_layer):
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False


def check_frozen_layers_peft_model(model):
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True' 
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        config: training configuration
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = (
            torch.version.cuda
            and torch.cuda.is_bf16_supported()
            and packaging.version.parse(torch.version.cuda).release >= (11, 0)
            and dist.is_nccl_available()
            and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy


def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be helpful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries, 
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (following FSDP checkpointing style) using properties of the train_config object

    save_dir = Path(train_config.dist_checkpoint_root_folder).joinpath(train_config.dist_checkpoint_folder).resolve()
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir, 'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank == 0:
            print(f"training params are saved in {file_name}")


def prepare_to_end_training(
        current_results: Dict[str, Any],
        epoch_times: List[float],
        checkpoint_times: List[float],
        train_prep: List[torch.Tensor],
        train_loss: List[float],
        val_prep: List[torch.Tensor],
        val_loss: List[torch.Tensor],
        train_config: TrainConfig,
        fsdp_config: FSDPConfig,
        rank: int
):
    results = update_results(
        current_results, epoch_times, checkpoint_times, train_prep, train_loss, val_prep, val_loss, train_config
    )

    # saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)

    return results
