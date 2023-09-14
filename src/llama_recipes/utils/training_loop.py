import os
import time

import torch
from torch import distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm

from llama_recipes.utils import MemoryTrace, gradient_update, run_validation, distributed_print, update_results, \
    prepare_to_end_training


def train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        lr_scheduler,
        gradient_accumulation_steps,
        train_config,
        fsdp_config=None,
        local_rank=None,
        rank=None,
):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    train_prep, train_loss = [], []
    val_prep, val_loss = [], []
    epoch_times, checkpoint_times = [], []
    results = {}
    current_patience = train_config.patience
    best_val_loss = float("inf")

    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length)
            for step, batch in enumerate(train_dataloader):
                for key in batch.keys():
                    if train_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    else:
                        batch[key] = batch[key].to('cuda:0')
                loss = model(**batch).loss
                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                gradient_update(
                    train_config,
                    scaler,
                    gradient_accumulation_steps,
                    train_dataloader,
                    optimizer,
                    pbar,
                    loss,
                    step
                )
                pbar.set_description(
                    f"Training Epoch: {epoch}/{train_config.num_epochs}, "
                    f"step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})"
                )
                if train_config.run_validation and train_config.validate_every_n_steps != -1:
                    if step != 0 and step % train_config.validate_every_n_steps == 0:
                        current_patience, checkpoint_times, val_loss, val_prep, best_val_loss, should_stop = run_validation(
                            model,
                            tokenizer,
                            optimizer,
                            train_config,
                            fsdp_config,
                            eval_dataloader,
                            local_rank,
                            rank,
                            epoch,
                            step,
                            best_val_loss,
                            val_loss,
                            val_prep,
                            checkpoint_times,
                            current_patience
                        )
                        if should_stop is True:
                            break

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        distributed_print(
            message=f"Max CUDA memory allocated was {memtrace.peak} GB\n" \
              f"Max CUDA memory reserved was {memtrace.max_reserved} GB\n" \
              f"Peak active CUDA memory was {memtrace.peak_active_gb} GB\n" \
              f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}\n" \
              f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB",
            rank=rank,
            fsdp=train_config.enable_fsdp
        )

        # end for early stopping mid-epoch
        if should_stop:
            results = prepare_to_end_training(
                results,
                epoch_times,
                checkpoint_times,
                train_prep,
                train_loss,
                val_prep,
                val_loss,
                train_config,
                fsdp_config,
                rank)
            return results

        # Update the learning rate as needed
        lr_scheduler.step()

        if train_config.run_validation and train_config.validate_every_n_steps == -1:
            current_patience, checkpoint_times, val_loss, val_prep, best_val_loss, should_stop = run_validation(
                model,
                tokenizer,
                optimizer,
                train_config,
                fsdp_config,
                eval_dataloader,
                local_rank,
                rank,
                epoch,
                step,
                best_val_loss,
                val_loss,
                val_prep,
                checkpoint_times,
                current_patience
            )

            # end for early stopping at the end of an epoch
            if should_stop:
                prepare_to_end_training(
                    results,
                    epoch_times,
                    checkpoint_times,
                    train_prep,
                    train_loss,
                    val_prep,
                    val_loss,
                    train_config,
                    fsdp_config,
                    rank)
                return results

            return results

        distributed_print(
            message=f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, "
                    f"train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s",
            rank=rank,
            fsdp=train_config.enable_fsdp
        )

    # end without early stopping
    prepare_to_end_training(
        results,
        epoch_times,
        checkpoint_times,
        train_prep,
        train_loss,
        val_prep,
        val_loss,
        train_config,
        fsdp_config,
        rank)

    return results

