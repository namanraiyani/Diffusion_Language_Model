"""
Inference Script for Masked Language Model (Diffusion-Style)
Supports both unconditional and conditional (Q&A) generation
"""

import sys
import os
import argparse
import logging
import torch
from transformers import AutoModelForMaskedLM
from rich.live import Live
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.text import Text
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.tokenizer import get_tokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)


def load_model_and_tokenizer(path_to_weights, hf_model_name, device="cuda"):
    tokenizer = get_tokenizer(hf_model_name)
    model = AutoModelForMaskedLM.from_pretrained(hf_model_name, device_map=device)
    model.resize_token_embeddings(len(tokenizer))

    print(f"Loading weights from {path_to_weights}")
    state_dict = load_file(path_to_weights)
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()
    model.eval()

    print("Model loaded ")
    return model, tokenizer


def prepare_unconditional_tokens(seq_len, mask_token_id, device="cuda"):
    input_tokens = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device) 
    return input_tokens, mask, attention_mask


def prepare_conditional_tokens(seq_len, tokenizer, prompt, device="cuda"):
    chat_template = [{"role": "user", "content": prompt}]

    tokenized = tokenizer.apply_chat_template(
        chat_template,
        tokenize=True,
        add_special_tokens=True,
        add_generation_prompt=True
    )

    prompt_tokens = torch.tensor(tokenized).to(device)
    input_tokens, mask, attention_mask = prepare_unconditional_tokens(
        seq_len, tokenizer.mask_token_id, device
    )

    input_tokens[0, :len(prompt_tokens)] = prompt_tokens
    mask[0, :len(prompt_tokens)] = False

    return input_tokens, mask, attention_mask


def format_display_qa(user_text, assistant_text):

    output = Text()
    output.append("USER: ", style="bold green")
    output.append(user_text + "\n\n")
    output.append("ASSISTANT: ", style="bold cyan")
    output.append(assistant_text, style="white")
    return output


def format_display_unconditional(gen_text):

    output = Text()
    output.append("Unconditional Generation: \n\n", style="bold green")
    output.append(gen_text, style="white")
    return output


def clean_text(raw_text):
    return raw_text.replace("user", "").replace("assistant", "").strip()


@torch.no_grad()
def inference(model, tokenizer, input_tokens, mask, attention_mask, num_steps, 
              remasking="random", device="cuda", prompt=None, show_mask=True):
    
    console = Console(highlight=False)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        
        task = progress.add_task("Generating...", total=num_steps)
        times = torch.linspace(1, 0, num_steps + 1, device=device)

        with Live("", refresh_per_second=5, console=console) as live:
            for t, s in zip(times[:-1], times[1:]):

                logits = model(input_tokens, attention_mask=attention_mask).logits
                probs = torch.softmax(logits[mask], dim=-1)
                input_tokens[mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)

                if remasking == "random":
                    remask_probs = torch.rand_like(mask, dtype=torch.float, device=device)
                    remask_probs = (remask_probs < s/t)
                    mask = mask & remask_probs
                    input_tokens[mask] = tokenizer.mask_token_id

                elif remasking == "low_confidence":
                    probs_all = torch.nn.functional.softmax(logits, dim=-1)
                    chosen_token_probs = torch.gather(probs_all, dim=-1, 
                                                      index=input_tokens.unsqueeze(-1)).squeeze(-1)
                    chosen_token_probs[~mask] = 1.0
                    num_to_remask = int((s/t) * mask.sum().item())

                    if num_to_remask > 0:
                        lowest_confidence_idx = torch.topk(chosen_token_probs, num_to_remask, 
                                                          largest=False).indices
                        new_mask = torch.zeros_like(mask)
                        new_mask[0, lowest_confidence_idx] = True
                        mask = new_mask
                        input_tokens[mask] = tokenizer.mask_token_id
                
                if show_mask:
                    decoded_tokens = tokenizer.convert_ids_to_tokens(input_tokens[0])
                    cleaned_tokens = []
                    for tok in decoded_tokens:
                        if tok == tokenizer.mask_token:
                            cleaned_tokens.append(tok)
                        elif tok in tokenizer.all_special_tokens:
                            continue
                        else:
                            cleaned_tokens.append(tok)
                    decoded_after = tokenizer.convert_tokens_to_string(cleaned_tokens)
                else:
                    decoded_after = tokenizer.batch_decode(input_tokens, skip_special_tokens=True)[0]

                if prompt is None:
                    format_text = format_display_unconditional(decoded_after)
                else:
                    assistant_text = decoded_after.replace(prompt, "").strip()
                    assistant_text = clean_text(assistant_text)
                    format_text = format_display_qa(prompt, assistant_text)
                
                live.update(format_text)
                progress.update(task, advance=1)

    console.print("\n" + "="*60)
    console.print("Generation Complete!", style="bold green")
    console.print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inference for Masked Language Model")
    
    parser.add_argument("--safetensors_path", required=True, type=str,
                       help="Path to model checkpoint (.safetensors file)")
    parser.add_argument("--hf_model_name", type=str, default="answerdotai/ModernBERT-base",
                       help="Huggingface model name for tokenizer")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda or cpu)")
    
    parser.add_argument("--prompt", type=str, default=None,
                       help="Prompt for conditional generation (Q&A mode)")
    parser.add_argument("--seq_len", type=int, default=256,
                       help="Maximum sequence length to generate")
    parser.add_argument("--num_steps", type=int, default=256,
                       help="Number of demasking steps")
    parser.add_argument("--strategy", type=str, default="random", 
                       choices=["random", "low_confidence"],
                       help="Remasking strategy")
    parser.add_argument("--show_mask", action="store_true",
                       help="Show [MASK] tokens during generation")

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        args.safetensors_path, 
        args.hf_model_name, 
        args.device
    )

    if args.prompt is None:
        print("\n" + "="*60)
        print("Unconditional Generation Mode")
        print("="*60 + "\n")
        
        input_tokens, mask, attention_mask = prepare_unconditional_tokens(
            args.seq_len, 
            mask_token_id=tokenizer.mask_token_id,
            device=args.device
        )
    else:
        print("\n" + "="*60)
        print("Conditional Generation Mode (Q&A)")
        print(f"Prompt: {args.prompt}")
        print("="*60 + "\n")
        
        input_tokens, mask, attention_mask = prepare_conditional_tokens(
            args.seq_len, 
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=args.device
        )

    inference(
        model,
        tokenizer,
        input_tokens, 
        mask, 
        attention_mask, 
        args.num_steps, 
        remasking=args.strategy, 
        device=args.device,
        prompt=args.prompt,
        show_mask=args.show_mask
    )


if __name__ == "__main__":
    main()


