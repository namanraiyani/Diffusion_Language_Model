
import os
import numpy as np
import torch


def get_model_size(model):
 
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6


def save_checkpoint(args, accelerator, completed_steps, progress_bar):
    path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
    path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_steps}")

    if accelerator.is_main_process:
        progress_bar.write(f"Saving Checkpoint to {path_to_checkpoint}")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.save_state(output_dir=path_to_checkpoint)


def setup_directories(args):

    path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
    os.makedirs(path_to_experiment, exist_ok=True)
    os.makedirs(os.path.join(path_to_experiment, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(path_to_experiment, "logs"), exist_ok=True)
    os.makedirs(os.path.join(path_to_experiment, "results"), exist_ok=True)
    return path_to_experiment


def count_model_parameters(model):

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params