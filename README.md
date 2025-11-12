# Diffusion_Language_Model

![vid9](https://github.com/user-attachments/assets/7a56e7f2-c282-4ec9-a333-9472f076fc7b)

Diffusion Language Models (DLMs) are a novel class of generative models that apply the diffusion process to discrete text generation. Unlike autoregressive models that generate tokens sequentially left-to-right, DLMs iteratively refine an initially masked or noisy sequence through a denoising process. The model starts with a sequence of completely masked tokens and gradually unmasks them over multiple diffusion steps, learning to predict which tokens should be revealed at each step. This approach draws inspiration from diffusion probabilistic models used in image generation but adapts them for discrete token spaces. The training objective involves randomly masking tokens at varying noise levels and teaching the model to recover the original text. During inference, the model performs iterative demasking, often using strategies like random remasking or confidence-based selection to determine which tokens to refine. DLMs offer several advantages including parallel generation capabilities, better handling of long-range dependencies, and the ability to perform conditional generation more naturally. They can also support bidirectional context modeling since they don't follow a strict left-to-right generation order. Recent implementations like MDLM (Masked Diffusion Language Models) have shown competitive performance with traditional autoregressive models while offering unique benefits for controlled generation and editing tasks. This paradigm represents a promising alternative to conventional language modeling approaches and continues to evolve with ongoing research.

```
# Install dependencies
pip install torch transformers datasets accelerate safetensors tokenizers tqdm rich

# Create directories
mkdir -p data experiments logs results
```

## Usage
Training
```
# Pretraining
python train.py \
    --experiment_name "my_exp" \
    --working_directory "./experiments" \
    --mode "pretrain" \
    --context_length 512 \
    --per_gpu_batch_size 8 \
    --num_training_steps 10000

# Fine-tuning (SFT)
python train.py \
    --experiment_name "sft_exp" \
    --working_directory "./experiments" \
    --mode "sft" \
    --path_to_pretrained_checkpoint "./experiments/my_exp/final_model/" \
    --num_training_steps 5000
```

Inference
```
# Unconditional generation
python inference.py \
    --safetensors_path "./experiments/my_exp/final_model/model.safetensors" \
    --seq_len 256 \
    --num_steps 256

# Q&A (conditional)
python inference.py \
    --safetensors_path "./experiments/my_exp/final_model/model.safetensors" \
    --prompt "What is machine learning?" \
    --seq_len 512

```

