import os
import sys
import json
import argparse
import torch
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.tokenizer import get_tokenizer
from src.utils import get_model_size


def compute_perplexity(model, tokenizer, dataset, max_samples=500, device="cuda"):
    model.eval()
    total_loss = 0
    total_tokens = 0
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    
    samples = min(max_samples, len(dataset))
    
    for i in tqdm(range(samples), desc="Computing Perplexity"):
        input_ids = torch.tensor([dataset[i]["input_ids"]], device=device)
        
        # Mask 15% of tokens
        mask_prob = 0.15
        mask = torch.rand(input_ids.shape, device=device) < mask_prob
        
        special_tokens = [tokenizer.pad_token_id, tokenizer.cls_token_id, 
                         tokenizer.sep_token_id, tokenizer.mask_token_id]
        for token_id in special_tokens:
            if token_id is not None:
                mask &= (input_ids != token_id)
        
        if mask.sum() == 0:
            continue
            
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = tokenizer.mask_token_id
        labels = input_ids.clone()
        labels[~mask] = -100
        
        try:
            outputs = model(input_ids=masked_input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()
        except:
            continue
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return float(perplexity)


def compute_mask_prediction_accuracy(model, tokenizer, dataset, max_samples=500, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    
    samples = min(max_samples, len(dataset))
    
    for i in tqdm(range(samples), desc="Computing Accuracy"):
        input_ids = torch.tensor([dataset[i]["input_ids"]], device=device)
        
        mask_prob = 0.15
        mask = torch.rand(input_ids.shape, device=device) < mask_prob
        
        special_tokens = [tokenizer.pad_token_id, tokenizer.cls_token_id, 
                         tokenizer.sep_token_id, tokenizer.mask_token_id]
        for token_id in special_tokens:
            if token_id is not None:
                mask &= (input_ids != token_id)
        
        if mask.sum() == 0:
            continue
            
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = tokenizer.mask_token_id
        
        try:
            logits = model(input_ids=masked_input_ids).logits
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions[mask] == input_ids[mask]).sum().item()
            total += mask.sum().item()
        except:
            continue
    
    if total == 0:
        return 0.0
    
    accuracy = correct / total
    return float(accuracy)


def compute_top_k_accuracy(model, tokenizer, dataset, k=5, max_samples=500, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    
    samples = min(max_samples, len(dataset))
    
    for i in tqdm(range(samples), desc=f"Computing Top-{k} Accuracy"):
        input_ids = torch.tensor([dataset[i]["input_ids"]], device=device)
        
        mask_prob = 0.15
        mask = torch.rand(input_ids.shape, device=device) < mask_prob
        
        special_tokens = [tokenizer.pad_token_id, tokenizer.cls_token_id, 
                         tokenizer.sep_token_id, tokenizer.mask_token_id]
        for token_id in special_tokens:
            if token_id is not None:
                mask &= (input_ids != token_id)
        
        if mask.sum() == 0:
            continue
            
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = tokenizer.mask_token_id
        
        try:
            logits = model(input_ids=masked_input_ids).logits
            top_k_predictions = torch.topk(logits, k=k, dim=-1).indices
            
            for pos in mask.nonzero(as_tuple=True)[1]:
                if input_ids[0, pos] in top_k_predictions[0, pos]:
                    correct += 1
                total += 1
        except:
            continue
    
    if total == 0:
        return 0.0
    
    accuracy = correct / total
    return float(accuracy)


def compute_inference_speed(model, tokenizer, seq_len=256, num_trials=50, device="cuda"):
    model.eval()
    
    dummy_input = torch.randint(0, len(tokenizer), (1, seq_len), device=device)
    for _ in range(5):
        _ = model(dummy_input)
    
    torch.cuda.synchronize()
    
    times = []
    for _ in range(num_trials):
        input_ids = torch.randint(0, len(tokenizer), (1, seq_len), device=device)
        
        start = time.time()
        _ = model(input_ids)
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)
    
    avg_time = float(np.mean(times))
    std_time = float(np.std(times))
    return avg_time, std_time


def evaluate_model(model, tokenizer, dataset, model_name, device="cuda", max_samples=500):
    """Evaluate a single model on all metrics"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}\n")
    
    results = {
        "model_name": model_name,
        "model_size_M": get_model_size(model),
        "evaluation_date": datetime.now().isoformat(),
        "num_samples": max_samples
    }
    
    print("Computing Perplexity...")
    results["perplexity"] = compute_perplexity(model, tokenizer, dataset, max_samples, device)
    
    print("Computing Mask Prediction Accuracy...")
    results["accuracy"] = compute_mask_prediction_accuracy(model, tokenizer, dataset, max_samples, device)
    
    print("Computing Top-5 Accuracy...")
    results["top5_accuracy"] = compute_top_k_accuracy(model, tokenizer, dataset, k=5, max_samples=max_samples, device=device)
    
    print("Computing Inference Speed...")
    avg_time, std_time = compute_inference_speed(model, tokenizer, seq_len=256, num_trials=50, device=device)
    results["inference_time_ms"] = avg_time
    results["inference_time_std_ms"] = std_time
    
    print(f"\nResults for {model_name}:")
    print(f"  Perplexity: {results['perplexity']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f}")
    print(f"  Inference Time: {results['inference_time_ms']:.2f} ± {results['inference_time_std_ms']:.2f} ms")
    print(f"  Model Size: {results['model_size_M']:.2f}M parameters")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Masked Language Models")
    
    parser.add_argument("--custom_checkpoint", type=str, required=True,
                       help="Path to your trained model checkpoint (.safetensors)")
    parser.add_argument("--custom_model_name", type=str, default="Custom Model",
                       help="Name for your custom model in results")
    parser.add_argument("--hf_model_name", type=str, default="answerdotai/ModernBERT-base",
                       help="Huggingface model name (for tokenizer)")
    
    parser.add_argument("--eval_data_path", type=str, default=None,
                       help="Path to evaluation dataset")
    parser.add_argument("--max_samples", type=int, default=500,
                       help="Maximum number of samples to evaluate")
    
    parser.add_argument("--baseline_models", type=str, nargs="+",
                       default=["bert-base-uncased", "roberta-base"],
                       help="Baseline models to compare")
    
    parser.add_argument("--output_path", type=str, default="./evaluation_results.json",
                       help="Path to save results")
    
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda or cpu)")
    
    args = parser.parse_args()
    
    print("Loading evaluation dataset...")
    if args.eval_data_path:
        dataset = load_from_disk(args.eval_data_path)
        if "test" in dataset:
            dataset = dataset["test"]
    else:
        checkpoint_dir = os.path.dirname(os.path.dirname(args.custom_checkpoint))
        data_path = os.path.join(checkpoint_dir, "data_pretrain")
        
        if os.path.exists(data_path):
            print(f"Found training data at {data_path}")
            dataset = load_from_disk(data_path)["test"]
        else:
            print("Downloading evaluation dataset...")
            dataset = load_dataset("manu/project_gutenberg", split="en")
            dataset = dataset.select(range(min(100, len(dataset))))
            
            tokenizer = get_tokenizer(args.hf_model_name)
            def tokenize(examples):
                return tokenizer(examples["text"], max_length=256, truncation=True)
            
            dataset = dataset.map(tokenize, remove_columns=["text"])
    
    print(f"Evaluation dataset size: {len(dataset)}")
    
    print("\n" + "="*60)
    print("EVALUATING CUSTOM MODEL")
    print("="*60)
    
    tokenizer = get_tokenizer(args.hf_model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.hf_model_name, device_map=args.device)
    model.resize_token_embeddings(len(tokenizer))
    
    state_dict = load_file(args.custom_checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()
    model.eval()
    
    custom_results = evaluate_model(model, tokenizer, dataset, args.custom_model_name, 
                                    args.device, args.max_samples)
    
    all_results = [custom_results]
    del model
    torch.cuda.empty_cache()
    
    output = {
        "evaluation_config": {
            "custom_checkpoint": args.custom_checkpoint,
            "max_samples": args.max_samples,
            "device": args.device,
        },
        "results": all_results
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()