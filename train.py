import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM, ModernBertForMaskedLM, 
    get_scheduler, AutoTokenizer
)
from datasets import load_from_disk
from accelerate import Accelerator
from tqdm import tqdm
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tokenizer import get_tokenizer
from src.data import prepare_pretraining_data, prepare_sft_data
from src.training import PretrainingCollator, SFTCollator, compute_mask_loss, compute_sft_loss
from src.utils import save_checkpoint, setup_directories, count_model_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Language Model")
    
    parser.add_argument("--experiment_name", required=True, type=str,
                       help="Name of the experiment")
    parser.add_argument("--working_directory", required=True, type=str,
                       help="Directory to save checkpoints and logs")
    parser.add_argument("--mode", type=str, default="pretrain", choices=["pretrain", "sft"],
                       help="Training mode: pretrain or sft (supervised fine-tuning)")
    

    parser.add_argument("--hf_model_name", type=str, default="answerdotai/ModernBERT-base",
                       help="Huggingface model name for tokenizer and base model")
    parser.add_argument("--path_to_pretrained_checkpoint", type=str, default=None,
                       help="Path to pretrained checkpoint (for SFT mode)")
    
    parser.add_argument("--path_to_prepped_data", type=str, default=None,
                       help="Path to prepared data. If None, will download and prepare automatically")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of books to use (default: all)")
    parser.add_argument("--context_length", type=int, default=1024,
                       help="Maximum context length for sequences")
    parser.add_argument("--large_dataset", action="store_true",
                       help="Use large-scale datasets (FineWeb + Wikipedia)")
    parser.add_argument("--test_split_pct", type=float, default=0.005,
                       help="Percentage of data for validation")
    parser.add_argument("--dataset_split_seed", type=int, default=42,
                       help="Random seed for dataset splitting")
    parser.add_argument("--batch_size", type=int, default=1000,
                       help="Batch size for dataset preprocessing")
    parser.add_argument("--huggingface_cache_dir", type=str, default=None,
                       help="Cache directory for HuggingFace datasets")
    
    parser.add_argument("--per_gpu_batch_size", type=int, default=16,
                       help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--num_training_steps", type=int, default=100000,
                       help="Total number of training steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for AdamW optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       choices=["linear", "cosine", "cosine_with_restarts", 
                               "polynomial", "constant", "constant_with_warmup"],
                       help="Learning rate scheduler type")
    parser.add_argument("--num_warmup_steps", type=int, default=1000,
                       help="Number of warmup steps")
    
    parser.add_argument("--logging_steps", type=int, default=1,
                       help="Log metrics every N steps")
    parser.add_argument("--evaluation_interval", type=int, default=2500,
                       help="Evaluate model every N steps")
    parser.add_argument("--checkpoint_interval", type=int, default=2500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--log_wandb", action="store_true",
                       help="Enable Weights & Biases logging")

    return parser.parse_args()


def train_pretraining(args, accelerator, model, tokenizer, train_dataloader, 
                     eval_dataloader, optimizer, scheduler):
    train = True
    completed_steps = 0
    progress_bar = tqdm(range(completed_steps, args.num_training_steps), 
                       disable=not accelerator.is_local_main_process)
    loss_func = nn.CrossEntropyLoss(reduction="none")

    while train:
        accumulate_steps = 0
        accumulate_loss = 0
        
        for batch in train_dataloader:        
            input_ids = batch["input_ids"].to(accelerator.device)
            batch_size, seq_len = input_ids.shape
            
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, 
                                       device=accelerator.device)
            t = torch.rand(batch_size, 1, device=accelerator.device).expand(
                batch_size, seq_len).clamp_min(1e-5)
            mask = torch.bernoulli(t).bool()

            masked_input_ids = input_ids.masked_fill(mask, tokenizer.mask_token_id)
            labels = input_ids.masked_fill(~mask, -100)

            logits = model(input_ids=masked_input_ids, attention_mask=attention_mask)["logits"]
            
            loss = compute_mask_loss(logits, labels, t, batch_size, seq_len)
            loss = loss / args.gradient_accumulation_steps
            accumulate_loss += loss

            accelerator.backward(loss)
            accumulate_steps += 1

            if accumulate_steps % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if completed_steps % args.logging_steps == 0:
                    accumulate_loss = accumulate_loss.detach()
                    if accelerator.state.num_processes > 1:
                        accumulate_loss = torch.mean(accelerator.gather_for_metrics(accumulate_loss))

                    log = {"train_loss": accumulate_loss.item(),
                           "learning_rate": scheduler.get_last_lr()[0]}

                    logging_string = f"[{completed_steps}/{args.num_training_steps}] Loss: {accumulate_loss:.4f}"
                    if accelerator.is_main_process:
                        progress_bar.write(logging_string)
                    if args.log_wandb:
                        accelerator.log(log, step=completed_steps)

                if completed_steps % args.evaluation_interval == 0:
                    evaluate_pretraining(args, accelerator, model, tokenizer, 
                                       eval_dataloader, loss_func, completed_steps)
                
                if (completed_steps % args.checkpoint_interval == 0) and completed_steps > 0:
                    save_checkpoint(args, accelerator, completed_steps, progress_bar)
                
                if completed_steps >= args.num_training_steps:
                    train = False
                    if accelerator.is_main_process:
                        progress_bar.write("Training Complete!")
                    break

                completed_steps += 1
                progress_bar.update(1)
                accumulate_loss = 0


def evaluate_pretraining(args, accelerator, model, tokenizer, eval_dataloader, 
                        loss_func, completed_steps):
    model.eval()
    total_loss = 0
    num_losses = 0

    for batch in tqdm(eval_dataloader, disable=not accelerator.is_main_process):
        input_ids = batch["input_ids"].to(accelerator.device)
        batch_size, seq_len = input_ids.shape
        
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, 
                                   device=accelerator.device)
        t = torch.rand(batch_size, 1, device=accelerator.device).expand(
            batch_size, seq_len).clamp_min(1e-5)
        mask = torch.bernoulli(t).bool()

        masked_input_ids = input_ids.masked_fill(mask, tokenizer.mask_token_id)
        labels = input_ids.masked_fill(~mask, -100)

        with torch.inference_mode():
            logits = model(input_ids=masked_input_ids, attention_mask=attention_mask)["logits"]
        
        loss = compute_mask_loss(logits, labels, t, batch_size, seq_len)
        loss = loss.detach()
        if accelerator.num_processes > 1:
            loss = torch.mean(accelerator.gather_for_metrics(loss))

        total_loss += loss
        num_losses += 1
    
    avg_loss = total_loss / num_losses
    if accelerator.is_main_process:
        print(f"[{completed_steps}] Validation Loss: {avg_loss:.4f}")
    if args.log_wandb:
        accelerator.log({"val_loss": avg_loss}, step=completed_steps)

    model.train()


def main():
    args = parse_args()
    path_to_experiment = setup_directories(args)

    accelerator = Accelerator(project_dir=path_to_experiment,
                             log_with="wandb" if args.log_wandb else None)
    if args.log_wandb:
        accelerator.init_trackers(args.experiment_name)

    tokenizer = get_tokenizer(args.hf_model_name)

    if args.path_to_prepped_data is None or not os.path.exists(args.path_to_prepped_data):
        if args.path_to_prepped_data is None:
            args.path_to_prepped_data = os.path.join(path_to_experiment, f"data_{args.mode}")
        
        if args.mode == "pretrain":
            tokenized_data = prepare_pretraining_data(args)
        else:
            tokenized_data = prepare_sft_data(args)
    else:
        print(f"Loading data from {args.path_to_prepped_data}")
        start = time.time()
        tokenized_data = load_from_disk(args.path_to_prepped_data)
        print(f"Data loaded in {time.time()-start:.2f}s")

    if args.mode == "pretrain":
        model = AutoModelForMaskedLM.from_pretrained(args.hf_model_name)
        model.resize_token_embeddings(len(tokenizer))
    else:
        if args.path_to_pretrained_checkpoint is None:
            raise ValueError("For SFT mode, --path_to_pretrained_checkpoint is required")
        
        model = ModernBertForMaskedLM.from_pretrained(args.hf_model_name)
        model.resize_token_embeddings(len(tokenizer))
        
        print(f"Loading pretrained weights from {args.path_to_pretrained_checkpoint}")
        state_dict = load_file(args.path_to_pretrained_checkpoint, device='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.tie_weights()

    params = count_model_parameters(model)
    accelerator.print(f"Parameters: {params:,}")

    mini_batchsize = args.per_gpu_batch_size // args.gradient_accumulation_steps
    collate_fn = PretrainingCollator() if args.mode == "pretrain" else SFTCollator(args.hf_model_name)

    train_dataloader = DataLoader(
        tokenized_data["train"], 
        batch_size=mini_batchsize,
        collate_fn=collate_fn, 
        shuffle=True
    )

    eval_dataloader = DataLoader(
        tokenized_data["test"], 
        batch_size=mini_batchsize,
        collate_fn=collate_fn, 
        shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, 
                                  weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.num_training_steps * accelerator.num_processes,
    )

    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    if args.mode == "pretrain":
        train_pretraining(args, accelerator, model, tokenizer, train_dataloader, 
                         eval_dataloader, optimizer, scheduler)

    path_to_checkpoint = os.path.join(path_to_experiment, "final_model")
    accelerator.save_state(output_dir=path_to_checkpoint)
    accelerator.end_training()

    print("Training complete!")



if __name__ == "__main__":
    main()


